# Attribution

`kinovsr/toflow/` is an independent MLX runtime for the released TOFlow
Torch7 checkpoints. The Torch7 module tree is converted into a static JSON graph
and interpreted by MLX-native operators; no upstream Lua code is bundled or
executed at runtime.

## TOFlow

Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T. Freeman --
"Video Enhancement with Task-Oriented Flow" (IJCV 2019).
https://github.com/anchen1011/toflow

Upstream license: MIT License, copyright (c) 2017 Baian Chen (Andrew).

## License Text

MIT License

Copyright (c) 2017 Baian Chen (Andrew)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
