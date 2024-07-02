use std::any::Any;

type Sender<D> = std::sync::mpsc::SyncSender<(
    NoSendCommand<D>,
    std::sync::mpsc::SyncSender<Box<dyn Any + Send>>,
)>;

pub struct NoSend<D>
where
    D: 'static,
{
    sender: Sender<D>,
}

struct NoSendCommand<D>(Box<dyn (FnOnce(&mut D) -> Box<dyn Any + Send>) + Send>);

impl<D> NoSend<D> {
    pub fn new(init: impl FnOnce() -> D + Send + 'static) -> Self {
        let (command_sender, command_receiver): (Sender<D>, _) = std::sync::mpsc::sync_channel(1);

        std::thread::spawn(move || {
            let mut data = init();
            loop {
                match command_receiver.recv() {
                    Ok((f, resp_sender)) => {
                        let Ok(()) = resp_sender.send((f.0)(&mut data)) else {
                            break;
                        };
                    }
                    Err(_) => break,
                }
            }
        });

        Self {
            sender: command_sender,
        }
    }
    pub fn with<E, F>(&self, f: F) -> E
    where
        F: FnOnce(&mut D) -> E + Send + 'static,
        E: Any + Send + 'static,
    {
        let (response_sender, response_receiver) = std::sync::mpsc::sync_channel(1);
        self.sender
            .send((
                NoSendCommand(Box::new(move |data| Box::new(f(data)))),
                response_sender,
            ))
            .expect("unreachable");

        let resp = response_receiver.recv().expect("unreachable");
        *resp.downcast().expect("unreachable")
    }
}
