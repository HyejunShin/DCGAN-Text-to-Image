# DCGAN-Text-to-Image

## Downloads
- [Oxford 102 flowers image dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [cvpr2016_flowers.tar.gz](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view?resourcekey=0-Av8zFbeDDvNcF1sSjDR32w)
- [flowers_icml.tar.gz](https://drive.google.com/file/d/0B0ywwgffWnLLMl9uOU91MV80cVU/view?resourcekey=0-vy4hVV-6Hvpw3AiYipoa3g)
- Run convert_flowers_to_hd5_script.py to obtain the flowers.hdf5 file

## Repository and Code Structure
```bash
DCGAN-Text-to-Image/
  |-- convert_flowers_to_hd5_script.py
  |-- trainer.py
  |-- runtime.py
  |-- txt2image_dataset.py
  |-- utils.py
  |-- flowers.hdf5
  |-- models
    |-- gan.py
    ......
  |-- 102flowers/
    |-- jpg/
        |-- image_00001.jpg
        |-- image_00002.jpg
        |-- image_00003.jpg
        ......
    |-- text_c10
        |-- class_00001
        |-- class_00002
        |-- class_00003
        ......
    |-- flowers_icml/
        |-- trainclasses.txt
        |-- valclasses.txt
        |-- testclasses.txt
        |-- class_00001
        |-- class_00002
        |-- class_00003
        ......
```

## Commands to Execute the Code
To run train():
```bash
python runtime.py
```

To run predict():
```bash
python runtime.py --inference --pre_trained_disc ./checkpoints/disc_190.pth --pre_trained_gen ./checkpoints/gen_190.pth
```

## Graphs


## Results


## References
1) [https://github.com/aelnouby/Text-to-Image-Synthesis](https://github.com/aelnouby/Text-to-Image-Synthesis)
2) [https://arxiv.org/pdf/1605.05396.pdf](https://arxiv.org/pdf/1605.05396.pdf)
3) 
