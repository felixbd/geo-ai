GEO AI
======

<!--
$$\\begin{array}{l} a = \\left\\{ \\begin{array}{ll} a, & \\mathrm{if} \\ a \\ge 0 \\\\ a \\cdot -1 + 90, & \\mathrm{otherwise} \\end{array} \\right. \\\\ b = \\left\\{ \\begin{array}{ll} b, & \\mathrm{if} \\ b \\ge 0 \\\\ b \\cdot -1 + 180, & \\mathrm{otherwise} \\end{array} \\right. \\\\ f(a, b) = \\mathopen{}\\left( \\mathrm{round} \\mathopen{}\\left( \\frac{a \\cdot \\mathrm{X\\_SCALE}}{180} \\mathclose{}\\right), \\mathrm{round} \\mathopen{}\\left( \\frac{b \\cdot \\mathrm{Y\\_SCALE}}{360} \\mathclose{}\\right) \\mathclose{}\\right) \\end{array}$$
-->

$$f(a, b) = \\mathopen{}\\left( \\mathrm{round} \\mathopen{}\\left( \\frac{\\left\\{ \\begin{array}{ll} a, & \\mathrm{if} \\ a \\ge 0 \\\\ a \\cdot -1 + 90, & \\mathrm{otherwise} \\end{array} \\right. \\cdot \\mathrm{X\\_SCALE}}{180} \\mathclose{}\\right), \\mathrm{round} \\mathopen{}\\left( \\frac{\\left\\{ \\begin{array}{ll} b, & \\mathrm{if} \\ b \\ge 0 \\\\ b \\cdot -1 + 180, & \\mathrm{otherwise} \\end{array} \\right. \\cdot \\mathrm{Y\\_SCALE}}{360} \\mathclose{}\\right) \\mathclose{}\\right)$$

<!--
$$\\begin{array}{l} a = \\frac{x \\cdot 180}{\\mathrm{X\\_SCALE}} \\\\ b = \\frac{y \\cdot 360}{\\mathrm{Y\\_SCALE}} \\\\ a = \\left\\{ \\begin{array}{ll} a, & \\mathrm{if} \\ a \\le 90 \\\\ \\mathopen{}\\left( a - 90 \\mathclose{}\\right) \\cdot -1, & \\mathrm{otherwise} \\end{array} \\right. \\\\ b = \\left\\{ \\begin{array}{ll} b, & \\mathrm{if} \\ b \\le 180 \\\\ \\mathopen{}\\left( b - 180 \\mathclose{}\\right) \\cdot -1, & \\mathrm{otherwise} \\end{array} \\right. \\\\ \\mathrm{inverse\\_f}(x, y) = \\mathopen{}\\left( a, b \\mathclose{}\\right) \\end{array}$$
-->

$$\\mathrm{inverse\\_f}(x, y) = \\mathopen{}\\left( \\left\\{ \\begin{array}{ll} \\frac{x \\cdot 180}{\\mathrm{X\\_SCALE}}, & \\mathrm{if} \\ \\frac{x \\cdot 180}{\\mathrm{X\\_SCALE}} \\le 90 \\\\ \\mathopen{}\\left( \\frac{x \\cdot 180}{\\mathrm{X\\_SCALE}} - 90 \\mathclose{}\\right) \\cdot -1, & \\mathrm{otherwise} \\end{array} \\right., \\left\\{ \\begin{array}{ll} \\frac{y \\cdot 360}{\\mathrm{Y\\_SCALE}}, & \\mathrm{if} \\ \\frac{y \\cdot 360}{\\mathrm{Y\\_SCALE}} \\le 180 \\\\ \\mathopen{}\\left( \\frac{y \\cdot 360}{\\mathrm{Y\\_SCALE}} - 180 \\mathclose{}\\right) \\cdot -1, & \\mathrm{otherwise} \\end{array} \\right. \\mathclose{}\\right)$$

<!--

## Notes:

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

TinyViT: Fast Pretraining Distillation for Small Vision Transformers

Reinforcement Learning

V-net (V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation)


# goal

pred land better than 96%

-->

## Street View Panoramas

> (188K Photospheres from Google Street View from 2007 to 2023)
>
> ### About Dataset
>
>The Streetview Panoramas Dataset is a comprehensive collection of 187,777 streetview photospheres from various locations around the world. This dataset provides researchers, developers, and enthusiasts with a diverse and extensive resource for a wide range of applications such as computer vision, geolocation analysis, urban planning, and virtual reality.
>
>The dataset consists of two main components: a CSV file named "images.csv" and a folder named "images." The CSV file serves as a metadata index for the photospheres, while the "images" folder contains the corresponding photospheres themselves. Each photosphere has a unique identifier (ID) assigned to it.
>
>The "images" folder contains the photospheres, each having a width of 512 pixels (and a variable height). The filenames of the photospheres correspond to their unique IDs, making it easy to establish a connection between the CSV metadata and the actual image files.
>
>This dataset provides an invaluable resource for various research and development tasks, including image classification, object recognition, geospatial analysis, and exploration of urban environments. With its vast collection of street view panoramas, researchers can explore diverse geographical locations and examine changes in urban landscapes over time.
>
> #### Google Streetview/Maps license
>
>https://www.google.com/intl/en-GB_ALL/permissions/geoguidelines/
>
>>
>> see: https://www.kaggle.com/datasets/nikitricky/streetview-photospheres/
>>
