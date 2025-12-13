import json
import math
import lmstudio as lms

def l2_distance(a, b):
    sum = 0
    for i in range(0, len(a)):
        difference = a[i] - b[i]
        sum += difference ** 2
    return math.sqrt(sum)

def cluster_distance_min(cluster_a, cluster_b):
    distances = []
    for a in cluster_a:
        for b in cluster_b:
            distances.append(l2_distance(a['embedding'], b['embedding']))
    return min(distances)

def cluster_distance_average(cluster_a, cluster_b):
    distances = []
    for a in cluster_a:
        for b in cluster_b:
            distances.append(l2_distance(a['embedding'], b['embedding']))
    return sum(distances) / len(distances)

def cluster_centroid(cluster):
    dimensions = len(cluster[0]['embedding'])
    current_centroid = [0.0] * dimensions
    for review in cluster:
        for i in range(dimensions):
            current_centroid[i] += review['embedding'][i]
    for i in range(dimensions):
        current_centroid[i] /= len(cluster)
    return current_centroid


def sum_square_distance(a, b):
    sum = 0
    for i in range(0, len(a)):
        difference = a[i] - b[i]
        sum += difference ** 2
    return sum

def cluster_distance_wards(cluster_a, cluster_b):
    centroid_a = cluster_centroid(cluster_a)
    centroid_b = cluster_centroid(cluster_b)
    distance_squared = sum_square_distance(centroid_a, centroid_b)
    return (len(cluster_a) * len(cluster_b)) / (len(cluster_a) + len(cluster_b)) * distance_squared

with open("reviews_embedding.json", "r") as f:
    data = json.load(f)

model = lms.embedding_model("text-embedding-qwen3-embedding-0.6b")

# Put every data point in a cluster by itself
clusters = []
for review_embedding in data:
    clusters.append([review_embedding])

while len(clusters) > 12:
    candidate_clusters = []

    min_distance = None
    closest_pair = None
    for i in range(0, len(clusters)):
        for j in range(i + 1, len(clusters)):
            distance = cluster_distance_wards(clusters[i], clusters[j])
            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest_pair = { "i": i, "j": j }
    
    merged_cluster = clusters[closest_pair['i']] + clusters[closest_pair['j']]
    new_clusters = [ merged_cluster ]
    for i in range(len(clusters)):
        if i == closest_pair['i'] or i == closest_pair['j']:
            continue
        new_clusters.append(clusters[i])
    
    clusters = new_clusters


for cluster in clusters:
    print("==== CLUSTER ===-")
    for review in cluster:
        print(review['review'])
