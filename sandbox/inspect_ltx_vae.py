#!/usr/bin/env python3

import inspect
from diffusers import AutoencoderKLLTXVideo

print("=" * 80)
print("Class:")
print(AutoencoderKLLTXVideo)

print("\n" + "=" * 80)
print("Module:")
print(AutoencoderKLLTXVideo.__module__)

print("\n" + "=" * 80)
print("Source file:")
print(inspect.getfile(AutoencoderKLLTXVideo))