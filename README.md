# Dreamfusion-with-CLIP-Cut



<p align="center">
  <img src="/demo/base.gif" alt="Vanilla Dreamfusion" width="300"/>
</p>
<p align="center">Vanilla Dreamfusion</p>

<br>
<br>

<p align="center">
  <img src="/demo/with%20clip%20cut.gif" alt="Dreamfusion With Clip Cut" width="300"/>
</p>
<p align="center">Dreamfusion With Clip Cut</p>

<br>
<br>

<p align="center">Prompt: "a DSLR photo of a corgi puppy"</p>

### Usage:



To test CLIP Cut, run:
```
!python main.py --text "a DSLR photo of a corgi puppy" --output/file -O
```


This phase is trained on a RTX 4090 GPU, takes around 60mins(Varies according to prompt used) 


The code is built based on [stable dreamfusion](https://github.com/ashawkey/stable-dreamfusion).
