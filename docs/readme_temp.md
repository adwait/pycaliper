#### The `config.json` file

The `config.json` file identifies configuration paths and options for backends (e.g., Jasper) and the design. It also identifies the specification file. More details on the specification file below.

## PyCaliper Specifications

PyCaliper specifications are Python classes inheriting from the abstract `Module` class. See examples in `specs/regblock.py`. The class declares signals in its `__init__` function. Further the `input`, `output` and `state` functions define input assumptions, output assertions and state invariants respectively.


### Jasper Backend

The Jasper backend serves as an alternative to the BTOR backend, and requires access to the Jasper FV App. 
We have tested PyCaliper with the Jasper FV App 2023.09 release. 

PyCaliper communicates with the Jasper FV App using a TCP socket interface.
To use the Jasper backed, launch the Jasper FV App with the `jasperserver.tcl` server script:

```
<path to jg> jasperserver.tcl
```
and *at the Jasper prompt* start a server (e.g., on port 8080)
```
jg_start_server 8080
```


