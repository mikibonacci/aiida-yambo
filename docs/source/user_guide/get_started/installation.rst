
Getting the plugin
------------------

The plugin can be installed from the Python Package Index (PyPI) using pip

::

    pip install aiida-yambo

or downloaded from the official github repository

::

    git clone https://github.com/yambo-code/yambo-aiida.git
    cd aiida_yambo
    pip install -e aiida-yambo

in order to successfully install the plugin, follow the relative AiiDA-core documentation.

Setup Yambo on AiiDA
---------------------

In order to set up the p2y and yambo executables as an AiiDA codes, use the name ``yambo.yambo`` as the plugin name

::

    $ verdi code setup
    At any prompt, type ? to get some help.
    ---------------------------------------
    => Label: yambo_codename
    => Description: YAMBO MBPT code
    => Local: False
    => Default input plugin: yambo.yambo
    => Remote computer name: @cluster
    => Remote absolute path: /your_path_to_yambo
    => Text to prepend to each command execution
    FOR INSTANCE, MODULES TO BE LOADED FOR THIS CODE:
       # This is a multiline input, press CTRL+D on a
       # empty line when you finish
       # ------------------------------------------
       # End of old input. You can keep adding
       # lines, or press CTRL+D to store this value
       # ------------------------------------------
    module load your_module_name
    => Text to append to each command execution:
       # This is a multiline input, press CTRL+D on a
       # empty line when you finish
       # ------------------------------------------
       # End of old input. You can keep adding
       # lines, or press CTRL+D to store this value
       # ------------------------------------------
    Code 'yambo_codename' successfully stored in DB.
    pk: 38316, uuid: 24f75bca-2975-49a5-af2f-97a917bd6ee4

To setup a code there is also the possibility to define a YAML-format file 

:: 

    ---
    label: "yambo.4.5"
    description: "yambo v4.5"
    input_plugin: "yambo.yambo"
    on_computer: true
    remote_abs_path: "path_to_yambo_folder/bin/yambo"
    computer: "@cluster"
    prepend_text: |
        ''module load ...
        export OMP_NUM_THREADS = ''

    append_text: ""

To store the code, just type ``verdi code setup --config file.yml``.

Tip: for SLURM schedulers, we suggest to set, in the computer(so, for all codes) or code(if you need case-sensitive) prepend text

::

    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 


This will automatically set the right number of threads. For PBS/Torque, you need to set the 
environment variable `OMP_NUM_THREADS `by using the custom_scheduler_commands in the options 
of the calculation.  