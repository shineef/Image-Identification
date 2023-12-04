import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# def extract_points(image_path, o_template_path, x_template_path, star_template_path):
def extract_points(image_path, o_template_path, x_template_path):
    # Load the image and the templates
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    o_template = cv2.imread(o_template_path, cv2.IMREAD_GRAYSCALE)
    x_template = cv2.imread(x_template_path, cv2.IMREAD_GRAYSCALE)
    # star_template = cv2.imread(star_template_path, cv2.IMREAD_GRAYSCALE)

    # Get the width and height of the templates
    o_w, o_h = o_template.shape[::-1]
    x_w, x_h = x_template.shape[::-1]
    # star_w, star_h = star_template.shape[::-1]

    # Use template matching to get the similarity maps
    o_result = cv2.matchTemplate(image, o_template, cv2.TM_CCOEFF_NORMED)
    x_result = cv2.matchTemplate(image, x_template, cv2.TM_CCOEFF_NORMED)
    # star_result = cv2.matchTemplate(image, star_template, cv2.TM_CCOEFF_NORMED)

    # Define a threshold for the similarity
    threshold = 0.55

    # Get the locations where the similarity exceeds the threshold
    o_locations = np.where(o_result >= threshold)
    x_locations = np.where(x_result >= threshold)
    # star_location = np.where(star_result >= threshold)

    star_threshold = 100

    # # Get the center points of the 'o's and 'x's
    o_points = [(pt[1] + o_w // 2, pt[0] + o_h // 2) for pt in zip(*o_locations[::-1])]
    x_points = [(pt[1] + x_w // 2, pt[0] + x_h // 2) for pt in zip(*x_locations[::-1])]
    # star = (star_location[1] + star_w // 2, star_location[0] + star_h // 2)

    x_points = [pt for pt in x_points if np.linalg.norm(np.array(pt) - np.array(star)) > star_threshold]

    o_points = np.array(o_points)
    x_points = np.array(x_points)

    # Group the 'o' points based on their proximity to each other
    o_clustering = DBSCAN(eps=3, min_samples=2).fit(o_points)
    o_points = np.array([np.mean(o_points[o_clustering.labels_ == i], axis=0) for i in range(max(o_clustering.labels_) + 1)])

    # Group the 'x' points based on their proximity to each other
    x_clustering = DBSCAN(eps=3, min_samples=2).fit(x_points)
    x_points = np.array([np.mean(x_points[x_clustering.labels_ == i], axis=0) for i in range(max(x_clustering.labels_) + 1)])

    # return np.array(o_points), np.array(x_points), star
    return np.array(o_points), np.array(x_points)

def extract_star(image_path, star_template_path):
    # Load the image and the template
    image = cv2.imread(image_path)
    star_template = cv2.imread(star_template_path, 0)

    # Get the dimensions of the template
    star_h, star_w = star_template.shape

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the matchTemplate function to find the 'star' locations
    star_result = cv2.matchTemplate(gray, star_template, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.9  # Increase the threshold

    # Get the locations where the similarity exceeds the threshold
    star_locations = np.where(star_result >= threshold)

    # Get the center points of the 'stars'
    star_points = [(pt[1] + star_w // 2, pt[0] + star_h // 2) for pt in zip(*star_locations[::-1])]

    # If multiple points are detected, return the point with the highest similarity
    if len(star_points) > 1:
        max_index = np.argmax(star_result[star_locations])
        star_points = [star_points[max_index]]

    return np.array(star_points)

# def extract_star(image_path):
#     # Load the image and convert it to grayscale
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Use a binary threshold to separate the shapes from the background
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#     # Find contours in the binary image
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         # Calculate the area and perimeter of the contour
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)

#         # Calculate the compactness (perimeter^2 / area)
#         compactness = perimeter ** 2 / area if area > 0 else float('inf')

#         # If the compactness is significantly different from that of a circle (4*pi), it's likely a star
#         if abs(compactness - 4 * np.pi) > 1.0:
#             # Calculate the centroid of the star
#             M = cv2.moments(contour)
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])

#             return (cx, cy)

#     return None

def classify_knn(o_points, x_points, star, k=3):
    # Calculate the distances to the star
    o_distances = distance.cdist(o_points, [star], 'euclidean')
    x_distances = distance.cdist(x_points, [star], 'euclidean')

    # Combine the distances and labels
    distances = np.concatenate((o_distances, x_distances))
    labels = np.array(['o'] * len(o_distances) + ['x'] * len(x_distances))

    # Get the k nearest neighbors
    neighbors = labels[np.argsort(distances, axis=0)[:k]]

    # The class of the star is the most common class among the neighbors
    return 'o' if np.count_nonzero(neighbors == 'o') > np.count_nonzero(neighbors == 'x') else 'x'

def classify_rocchio(o_points, x_points, star):
    # Calculate the centroids of the 'o' and 'x' points
    o_centroid = np.mean(o_points, axis=0)
    x_centroid = np.mean(x_points, axis=0)

    # The class of the star is the class whose centroid is closest to it
    return 'o' if distance.euclidean(star, o_centroid) < distance.euclidean(star, x_centroid) else 'x'

# Replace these paths with the paths to your image and templates
image_path = 'points.png'
o_template_path = 'o.png'
x_template_path = 'x.png'
star_template_path = 'star.png'
star = extract_star(image_path, star_template_path)
star = np.squeeze(star)
# print(star)

o_points, x_points = extract_points(image_path, o_template_path, x_template_path)

# o_points, x_points, star = extract_points(image_path, o_template_path, x_template_path, star_template_path)

# Plot the points
plt.figure(figsize=(8, 8))
plt.scatter(o_points[:, 0], -o_points[:, 1], color='blue', label='o')
plt.scatter(x_points[:, 0], -x_points[:, 1], color='red', label='x')
plt.scatter(star[0], -star[1], color='green', label='star')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

print(f'The location of the o points is at: {o_points}')
print(f'The location of the x points is at: {x_points}')
print(f'The location of the star is at: {star}')
print(f'k-NN classification: {classify_knn(o_points, x_points, star)}')
print(f'Rocchio classification: {classify_rocchio(o_points, x_points, star)}')