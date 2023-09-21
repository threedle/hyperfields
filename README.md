# HyperFields
The offcial website for HyperFields
# HyperFields: Towards Zero-Shot Generation of NeRFs from Text

[Sudarshan babu*<sup>&dagger;</sup>](https://people.cs.uchicago.edu/~sudarshan/), [Richard Liu*<sup>&ddagger;</sup>](https://rgliu.com/), [Avery Zhou*<sup>&dagger;&ddagger;</sup>](https://github.com/AveryZhou), [Michael Maire <sup>&ddagger;</sup>](https://people.cs.uchicago.edu/~mmaire/), [Gregory Shakhnarovich <sup>&dagger;</sup>](https://home.ttic.edu/~gregory/), [Rana Hanocka <sup>&ddagger;</sup>](https://people.cs.uchicago.edu/~ranahanocka/) 

&dagger; Toyota Technological Institute at Chicago, &ddagger; University of Chicago, * equal contribution

Abstract: We introduce HyperFields, a method for generating text-conditioned NeRFs with a single forward pass and (optionally) some finetuning. Key to our approach are: (i) a dynamic hypernetwork, which learns a smooth mapping from text token embeddings to the space of Neural Radiance Fields (NeRFs); (ii) NeRF distillation training, which distills scenes encoded in individual NeRFs into one dynamic hypernetwork. These techniques enable a single network to fit over a hundred unique scenes. We further demonstrate that HyperFields learns a more general map between text and NeRFs, and consequently is capable of predicting novel in-distribution and out-of-distribution scenes --- either zero-shot or with a few finetuning steps. Finetuning HyperFields benefits from accelerated convergence thanks to the learned general map, and is capable of synthesizing novel scenes 5 to 10 times faster than existing neural optimization-based methods. Our ablation experiments show that both the dynamic architecture and NeRF distillation are critical to the expressivity of HyperFields.






