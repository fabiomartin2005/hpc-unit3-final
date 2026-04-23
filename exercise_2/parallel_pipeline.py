from multiprocessing import Pool
import numpy as np
from skimage.measure import label, regionprops
import time

def process_image(img):
    binary = img > 0.5
    labeled = label(binary)
    props = regionprops(labeled)

    areas = [p.area for p in props]

    return len(props)


def run(workers=4):
    np.random.seed(42)

    images = [np.random.rand(256, 256) for _ in range(20)]

    start = time.time()

    with Pool(workers) as p:
        results = p.map(process_image, images)

    end = time.time()

    print(f"[PARALLEL] workers={workers}, time:", end - start)
    return end - start


if __name__ == "__main__":
    run()