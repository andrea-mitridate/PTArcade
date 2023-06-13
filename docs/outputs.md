# Output
The structure of PTArcade outptut is the following

=== "ENTERPRISE mode"
    When running PTArcade in ENTERPRISE mode (`mod = enterprise` 
    in the [configuration file][config]), the structure of PTArcade
    output is the following:
    ```{ .sh .no-copy }
    out_dir/ # (1)!
    └── name/ # (2)!
        └── chain_0/
            ├── chain_1.txt
            ├── pars.txt
            ├── prior.txt
            ├── runtime_info.txt
            └── ...
    ```

    1.  By default `out_dir = ./chains`. The user can specify a different 
    output directory via the [configuration file][config].
    2.  By default `name=np_model`. The user can specify a different 
    name in the [model file][model].

=== "ceffyl mode"
    When running PTArcade in ENTERPRISE mode (`mode = ceffyle` 
    in the [configuration file][config]), the structure of PTArcade
    output is the following:
    ```{ .sh .no-copy }
    out_dir/ # (1)!
    └── name/ # (2)!
        └── chain_0/
            ├── chain_1.txt
            ├── pars.txt
            ├── prior.txt
            ├── runtime_info.txt
            └── ...
    ```

    1.  By default `out_dir = ./chains`. The user can specify a different 
    output directory via the [configuration file][config].
    2.  By default `name=np_model`. The user can specify a different 
    name in the [model file][model].

[config]: ./inputs/config.md
[model]: ./inputs/model.md