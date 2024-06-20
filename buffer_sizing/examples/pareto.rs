use grb::Expr;
use sdf3_xml_parser::parse;
use grb::prelude::*;

fn main() -> anyhow::Result<()> {
    let Some(file) = std::env::args().nth(1) else {
        std::process::exit(1);
    };

    let s = std::fs::read_to_string(file)?;

    let (mut milp, execution_time, buffers) = parse(&s);

    println!("Parsed");

    let throughput = &milp.throughputs[0];

    let buffer_size = buffers.iter().sum::<Expr>();
    
    let mut cycle_time_ub = milp.hsdf.repetition_vector.iter().enumerate().map(|(i, r)| r[0] * execution_time.get(&i).unwrap()).sum::<usize>();
    
    let model = &mut milp.model;
    let mut cycle_time_constraint= model.add_constr("no_deadlock", grb::c!(throughput.clone()*cycle_time_ub >= 1))?;
    let mut capacity_constraint= model.add_constr("capacity", c!(buffer_size.clone() <= grb::INFINITY))?;
    
    model.get_env_mut().set(grb::param::LogToConsole, 0)?;
    //very numerically unstable optimization
    model.get_env_mut().set(grb::param::IntFeasTol, 1e-9)?;
    model.get_env_mut().set(grb::param::FeasibilityTol, 1e-9)?;
    model.get_env_mut().set(grb::param::OptimalityTol, 1e-9)?;

    loop {
        model.set_objective(0.0*throughput.clone() + buffer_size.clone(), grb::ModelSense::Minimize)?;
        model.remove(capacity_constraint)?;
        model.optimize()?;
        
        if model.status()? != Status::Optimal {
            break;
        }
        let capacity: usize = model.get_obj_attr_batch(attr::X, buffers.clone())?.iter().map(|e| e.round() as usize).sum();
        capacity_constraint = model.add_constr("capacity", c!(buffer_size.clone() <= capacity))?;
        model.set_objective(cycle_time_ub*throughput.clone() + 0.0*buffer_size.clone(), grb::ModelSense::Maximize)?;
        model.remove(cycle_time_constraint)?;
        model.optimize()?;
        if model.status()? != Status::Optimal {
            break;
        }
        let cycle_time = (1.0/model.get_obj_attr(attr::X, throughput)?).round() as usize;
        cycle_time_ub = cycle_time - 1;
        cycle_time_constraint = model.add_constr("no_deadlock", grb::c!(throughput.clone()*cycle_time_ub >= 1))?;

        println!("Pareto: {} {}", cycle_time, capacity);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() -> grb::Result<()> {
        use grb::prelude::*;

        let mut model = Model::new("model")?;

        let x = add_ctsvar!(model, name: "x", bounds: 0..)?;
        let y = add_ctsvar!(model, name: "y", bounds: 0..)?;

        model.add_constr("c", c!(x + y <= 10))?;

        model.set_objective(2*x, ModelSense::Maximize)?;
        model.optimize()?;

        println!("x:{}\ty:{}", model.get_obj_attr(attr::X, &x)?, model.get_obj_attr(attr::X, &y)?);

        model.set_objective(y, ModelSense::Maximize)?;
        model.optimize()?;

        println!("x:{}\ty:{}", model.get_obj_attr(attr::X, &x)?, model.get_obj_attr(attr::X, &y)?);

        Ok(())
    }
}
