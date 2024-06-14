#![feature(iterator_try_collect)]

use std::collections::BTreeMap;

use buffer_sizing::BufferedMrsdf;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sdf3 {
    application_graph: ApplicationGraph,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApplicationGraph {
    sdf: Sdf,
    sdf_properties: SdfProperties,
}

#[derive(Debug, Serialize, Deserialize)]
struct Sdf {
    #[serde(rename = "$value")]
    actors_or_channel: Vec<ActorOrChannel>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Actor {
    name: String,
    #[serde(rename = "$value")]
    ports: Vec<Port>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Port {
    name: String,
    #[serde(rename = "type")]
    t: String,
    rate: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Channel {
    name: String,
    src_actor: String,
    src_port: String,
    dst_actor: String,
    dst_port: String,
    initial_tokens: Option<isize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SdfProperties {
    #[serde(rename = "$value")]
    properties: Vec<ActorOrChannelProperties>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ActorOrChannel {
    Actor(Actor),
    Channel(Channel),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
enum ActorOrChannelProperties {
    ActorProperties(ActorProperties),
    ChannelProperties(ChannelProperties),
    GraphProperties,
}

#[derive(Debug, Serialize, Deserialize)]
struct ActorProperties {
    #[serde(rename = "$value")]
    processors: Vec<Processor>,
    actor: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Processor {
    #[serde(rename = "type")]
    t: String,
    default: bool,
    #[serde(rename = "$value")]
    execution_times: Vec<ExecutionTime>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExecutionTime {
    time: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChannelProperties {}

pub fn parse(
    s: &str,
) -> (
    milp_formulation::MilpFormulation<
        'static,
        1,
        impl milp_formulation::ExecutionTimeT<1>,
        impl milp_formulation::NameT<1>,
    >,
    BTreeMap<usize, usize>,
    Vec<grb::Var>,
) {
    use serde_xml_rs::from_str;
    use std::borrow::Cow;
    use std::collections::BTreeMap;

    let sdf: Sdf3 = from_str(s).unwrap();
    let ApplicationGraph {
        sdf,
        sdf_properties,
    } = &sdf.application_graph;
    let actors = sdf
        .actors_or_channel
        .iter()
        .filter_map(|e| {
            if let ActorOrChannel::Actor(a) = e {
                Some(a)
            } else {
                None
            }
        })
        .collect::<Vec<&Actor>>();
    let channels = sdf
        .actors_or_channel
        .iter()
        .filter_map(|e| {
            if let ActorOrChannel::Channel(c) = e {
                Some(c)
            } else {
                None
            }
        })
        .collect::<Vec<&Channel>>();
    let ports: BTreeMap<(&str, &str), &Port> = actors
        .iter()
        .flat_map(|a| a.ports.iter().map(|p| ((a.name.as_str(), p.name.as_str()), p)))
        .collect();
    let actor_indicies: BTreeMap<&str, usize> = actors
        .iter()
        .enumerate()
        .map(|(i, a)| (a.name.as_str(), i))
        .collect();
    let names = actor_indicies.iter().map(|(n, i)| (*i, n.to_string())).collect::<BTreeMap<_, _>>();
    let mut result = mdsdf::Mdsdf::<1>::new(actors.len());

    let execution_times: BTreeMap<usize, usize> = sdf_properties
        .properties
        .iter()
        .filter_map(|e| {
            if let ActorOrChannelProperties::ActorProperties(a) = e {
                Some(a)
            } else {
                None
            }
        })
        .map(|p| {
            (
                *actor_indicies.get(p.actor.as_str()).unwrap(),
                p.processors
                    .iter()
                    .filter(|e| e.default)
                    .nth(0)
                    .unwrap()
                    .execution_times[0]
                    .time,
            )
        })
        .collect();
    
    let mut channels_to_buffer = Vec::new();
    for Channel {
        name,
        src_actor,
        src_port,
        dst_actor,
        dst_port,
        initial_tokens,
    } in channels.iter()
    {
        if let Some(initial_tokens) = initial_tokens {
            result.add_channel(mdsdf::Channel {
                production_rate: [ports.get(&(src_actor.as_str(), src_port.as_str())).unwrap().rate].into(),
                consumption_rate: [ports.get(&(dst_actor.as_str(), dst_port.as_str())).unwrap().rate].into(),
                source: *actor_indicies.get(src_actor.as_str()).unwrap(),
                target: *actor_indicies.get(dst_actor.as_str()).unwrap(),
                initial_tokens: [*initial_tokens].into(),
            });
        } else {
            let channel = mdsdf::Channel {
                production_rate: [ports.get(&(src_actor.as_str(), src_port.as_str())).unwrap().rate].into(),
                consumption_rate: [ports.get(&(dst_actor.as_str(), dst_port.as_str())).unwrap().rate].into(),
                source: *actor_indicies.get(src_actor.as_str()).unwrap(),
                target: *actor_indicies.get(dst_actor.as_str()).unwrap(),
                initial_tokens: [0isize].into(),
            };
            channels_to_buffer.push((result.add_channel(channel), name));
        }
    }

    let mut milp = milp_formulation::MilpFormulation::new(
        Cow::Owned(result.clone().into_hsdf().into()),
        {
            let execution_times = execution_times.clone();
            move |(i, _)| *execution_times.get(&i).unwrap()
        },
        move |(i, j)| format!("{}({})", names.get(&i).unwrap(), j[0]),
    )
    .unwrap();

    let model = &mut milp.model;
    let buffers = channels_to_buffer.iter().map(|(_, name)| grb::add_ctsvar!(model, name: name, bounds: 0..)).try_collect::<Vec<_>>().unwrap();

    let mut buffer_sizing = BufferedMrsdf::new(&mut milp);
    for ((channel, _), buffer_size) in channels_to_buffer.iter().zip(buffers.iter()) {
        buffer_sizing.add_buffer(*channel, [buffer_size.into()].into()).unwrap();
    }

    (milp, execution_times, buffers)
}
