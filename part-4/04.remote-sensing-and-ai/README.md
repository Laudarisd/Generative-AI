# Remote Sensing and AI

Remote sensing uses data collected from a distance, typically by satellites, aircraft, drones, or radar systems. AI helps convert that raw data into forecasts, maps, classifications, detections, and scientific insight.

## 1. What Remote Sensing Data Looks Like

Common modalities include:

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
- geospatially structured

AI is used for:

- land cover classification
- change detection
- object detection
- crop monitoring
- disaster assessment
- climate and environmental forecasting
- flood and wildfire analysis

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

## 4. Why Remote Sensing Is Not Just Ordinary Computer Vision

Remote sensing data differs from internet images because it often includes:

- non-RGB channels
- physical sensor calibration issues
- geolocation constraints
- repeated observations over time
- large spatial scenes

That means domain knowledge matters much more.

## 5. Challenges Specific to Remote Sensing

- clouds and occlusion
- geospatial alignment
- domain shift across sensors
- limited labels
- class imbalance
- very large image size
- temporal gaps in observations

## 6. Typical Model Families

- CNNs for spatial patterns
- transformers for long-range spatial context
- temporal models for multi-date imagery
- multimodal fusion models
- self-supervised pretraining on satellite imagery
- segmentation architectures such as UNet-like models

## 7. Example: Simple NDVI Computation

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

## 8. AI Workflow for Remote Sensing

Typical pipeline:

1. acquire and preprocess imagery
2. correct or normalize sensor effects
3. align geospatial layers
4. prepare labels
5. train model
6. evaluate spatially and temporally
7. deploy or map outputs

## 9. Why Evaluation Is Hard

A model may perform well on one region, season, or sensor but fail on another.

That means evaluation should consider:

- geography shift
- season shift
- sensor shift
- temporal generalization
- rare event detection

## 10. Why Math Still Matters

Remote sensing is not only image classification.

It requires:

- geometry
- radiometry
- signal processing
- statistics
- optimization
- uncertainty handling

## 11. Practical Examples of Tasks

### Land Cover Classification

Assign each image or pixel to categories such as:

- forest
- urban
- water
- cropland

### Change Detection

Compare two or more dates to detect changes such as:

- deforestation
- flood damage
- urban growth
- wildfire impact

### Segmentation

Predict a label for each pixel or region.

## 12. Tiny Classification Example

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.array([
    [0.2, 0.7],
    [0.3, 0.8],
    [0.8, 0.2],
    [0.75, 0.25],
])
y = np.array([0, 0, 1, 1])

model = RandomForestClassifier(random_state=0)
model.fit(X, y)
print(model.predict([[0.25, 0.75]]))
```

## Problems to Think About

1. Why is domain shift common in satellite models?
2. What makes hyperspectral data different from RGB imagery?
3. Why can cloud cover create label and data problems?
4. Why is remote sensing often a spatiotemporal problem rather than a pure image problem?
5. Why do physical indices such as NDVI still matter in AI pipelines?

## Summary

Remote sensing and AI sit at the intersection of vision, signal processing, Earth science, and geospatial modeling. Strong systems in this area combine machine learning with sensor knowledge, spatial reasoning, and careful evaluation across time and geography.
