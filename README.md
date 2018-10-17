# Standalone gpustat

This is a standalone script of `gpustat`. You can find the official version in
[https://github.com/wookayin/gpustat](https://github.com/wookayin/gpustat).


# Dependencies

Python 3


# Execution

`path/to/gpustat.py`

If you put the script in your home directory for example, you can `watch` it by
running:

```bash
watch --color -n 1 -t "python ~/gpustat.py -u -c 2> /dev/null"
```

which will execute the command once per second, will hide `watch`'s header, and
will ignore any errors printed by the script. The flag `-u` will display the
users with active processes in each card, and `-c` will display the processes.
