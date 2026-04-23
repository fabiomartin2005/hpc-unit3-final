import numpy as np
from skimage.measure import label, regionprops
import time

def process_image(img):
    # binarizar (simulación de segmentación)
    binary = img > 0.5

    labeled = label(binary)
    props = regionprops(labeled)

    areas = [p.area for p in props]
    major = [p.major_axis_length for p in props]
    minor = [p.minor_axis_length for p in props]

    return {
        "cells": len(props),
        "avg_area": np.mean(areas) if areas else 0,
        "avg_major": np.mean(major) if major else 0,
        "avg_minor": np.mean(minor) if minor else 0
    }


def run():
    np.random.seed(42)

    images = [np.random.rand(256, 256) for _ in range(20)]

    start = time.time()

    results = []
    for img in images:
        results.append(process_image(img))

    end = time.time()

    print("[SERIAL] time:", end - start)
    return end - start


if __name__ == "__main__":
    run()