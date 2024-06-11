#![feature(exit_status_error)]

use npm_rs::*;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=plotter/src/index.ts");

    NpmEnv::default()
        .with_node_env(&NodeEnv::Development)
        .set_path(std::env::current_dir().unwrap().join("plotter"))
        .init_env()
        .install(None)
        .run("build")
        .exec()
        .unwrap()
        .exit_ok()
        .unwrap();
}
