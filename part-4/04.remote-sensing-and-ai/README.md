# Remote Sensing and AI

Remote sensing uses data collected from a distance, typically by satellites, aircraft, drones, or radar systems. AI helps convert that raw data into forecasts, maps, classifications, detections, and scientific insight.

## 1. What Remote Sensing Data Looks Like

Common modalities:

- optical imagery
- multispectral imagery
- hyperspectral imagery
- SAR, or synthetic aperture radar
- LiDAR
- thermal imagery
- temporal image stacks

Each modality has different noise, geometry, and physical meaning.

## 2. Why AI Helps

Remote sensing problems are often:

- high dimensional
- spatial
- temporal
- noisy
- weakly labeled

AI is used for:

- land cover classification
- change detection
- object detection
- crop monitoring
- disaster assessment
- climate and environmental forecasting

## 3. Core Mathematical View

If $X \in \mathbb{R}^{H \times W \times C}$ is an image cube, then a model learns:

```math
\hat{y} = f_\theta(X)
```

Depending on the task:

- $\hat{y}$ may be a class
- a segmentation mask
- a bounding box set
- a future geophysical estimate

## 4. Challenges Specific to Remote Sensing

- clouds and occlusion
- geospatial alignment
- domain shift across sensors
- limited labels
- class imbalance
- very large image size

## 5. Typical Model Families

- CNNs for spatial patterns
- transformers for long-range spatial context
- temporal models for multi-date imagery
- multimodal fusion models
- self-supervised pretraining on satellite imagery

## 6. Example: Simple NDVI Computation

One classic vegetation index is NDVI:

```math
\mathrm{NDVI} = \frac{NIR - Red}{NIR + Red}
```

This is not a neural model, but it shows how domain knowledge and AI pipelines often work together.

### Python Example

```python
import numpy as np

nir = np.array([[0.7, 0.8], [0.6, 0.5]])
red = np.array([[0.2, 0.3], [0.3, 0.2]])

ndvi = (nir - red) / (nir + red + 1e-9)
print(ndvi)
```

## 7. AI Workflow for Remote Sensing

Typical pipeline:

1. acquire and preprocess imagery
2. correct or normalize sensor effects
3. align geospatial layers
4. prepare labels
5. train model
6. evaluate spatially and temporally
7. deploy or map outputs

## 8. Why Math Still Matters

Remote sensing is not only image classification.

It requires:

- geometry
- radiometry
- signal processing
- statistics
- optimization
- uncertainty handling

## Problems to Think About

1. Why is domain shift common in satellite models?
2. What makes hyperspectral data different from RGB imagery?
3. Why is geospatial alignment critical?
4. How can a simple index like NDVI complement AI models?
5. What makes evaluation harder in remote sensing than in ordinary image tasks?

## References

- NASA Earthdata overview for remote sensing context: https://www.earthdata.nasa.gov/learn/backgrounders/remote-sensing
