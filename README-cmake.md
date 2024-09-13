changelog

- initial cmake support
- CLI supports relative-to-range mode.

recommended setup

- Basically Ubuntu-default GCC
    - 13 for 24.04
    - 11 for 22.04
    - 9 for 20.04
- CUDA (>= 11.4)

compilation instruction

```bash
## keep adding CUDA arch string for your GPU
## P100 -> 60  
## V100 -> 70   RTX20 -> 75  
## A100 -> 80   RTX30 -> 86   RTX40 -> 89  
## H100 -> 90

mkdir build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -DCMAKE_BUILD_TYPE=Release
```

CLI

```
    1      2  3  4  5   6
fz  fname  x  y  z  eb  use_rel[yes|no]
```


