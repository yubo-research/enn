use rand::Rng;

pub fn kmeans_plus_plus_init(
    points: &[Vec<f32>],
    k: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<f32>> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(points.len());
    let mut centroids = vec![points[rng.gen_range(0..points.len())].clone()];
    while centroids.len() < k {
        let mut dists = Vec::with_capacity(points.len());
        for p in points {
            let d = centroids
                .iter()
                .map(|c| crate::distance::l2_sq_f32(p, c))
                .fold(f32::INFINITY, f32::min);
            dists.push(d);
        }
        let sum: f32 = dists.iter().sum();
        if sum <= 0.0 {
            centroids.push(points[rng.gen_range(0..points.len())].clone());
            continue;
        }
        let mut pick = rng.gen::<f32>() * sum;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            pick -= d;
            if pick <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(points[chosen].clone());
    }
    centroids
}

pub fn assign_clusters(points: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    points
        .iter()
        .map(|p| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    crate::distance::l2_sq_f32(p, a)
                        .partial_cmp(&crate::distance::l2_sq_f32(p, b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

pub fn recompute_centroids(
    points: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
) -> Vec<Vec<f32>> {
    let dim = points.first().map_or(0, |p| p.len());
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (p, &a) in points.iter().zip(assignments.iter()) {
        if a < k {
            counts[a] += 1;
            for (j, &v) in p.iter().enumerate() {
                sums[a][j] += v;
            }
        }
    }
    (0..k)
        .map(|c| {
            if counts[c] == 0 {
                vec![0.0; dim]
            } else {
                sums[c]
                    .iter()
                    .map(|&s| s / counts[c] as f32)
                    .collect()
            }
        })
        .collect()
}

pub fn kmeans_run(
    points: &[Vec<f32>],
    k: usize,
    max_iters: usize,
    rng: &mut impl Rng,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let k = k.min(points.len()).max(1);
    let mut centroids = kmeans_plus_plus_init(points, k, rng);
    let mut assignments = assign_clusters(points, &centroids);
    for _ in 0..max_iters {
        let new_centroids = recompute_centroids(points, &assignments, k);
        let new_assignments = assign_clusters(points, &new_centroids);
        if new_assignments == assignments {
            centroids = new_centroids;
            break;
        }
        centroids = new_centroids;
        assignments = new_assignments;
    }
    (centroids, assignments)
}

pub struct PartitionTree {
    pub leaf_capacity: usize,
    pub root: PartitionNode,
}

pub enum PartitionNode {
    Leaf {
        entries: Vec<(u32, Vec<f32>)>,
        centroid: Vec<f32>,
    },
    Internal {
        centroid: Vec<f32>,
        children: Vec<PartitionNode>,
    },
}

impl PartitionTree {
    pub fn build(row_ids: &[u32], vectors: &[Vec<f32>], leaf_capacity: usize, seed: u64) -> Self {
        assert_eq!(row_ids.len(), vectors.len());
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let root = partition_recursive(row_ids, vectors, leaf_capacity, &mut rng);
        Self {
            leaf_capacity,
            root,
        }
    }

    pub fn all_leaves(&self) -> Vec<(Vec<u32>, Vec<f32>)> {
        let mut out = Vec::new();
        collect_leaves(&self.root, &mut out);
        out
    }
}

fn collect_leaves(node: &PartitionNode, out: &mut Vec<(Vec<u32>, Vec<f32>)>) {
    match node {
        PartitionNode::Leaf { entries, centroid } => {
            let ids: Vec<u32> = entries.iter().map(|(id, _)| *id).collect();
            out.push((ids, centroid.clone()));
        }
        PartitionNode::Internal { children, .. } => {
            for c in children {
                collect_leaves(c, out);
            }
        }
    }
}

fn partition_recursive(
    row_ids: &[u32],
    vectors: &[Vec<f32>],
    leaf_capacity: usize,
    rng: &mut impl Rng,
) -> PartitionNode {
    partition_recursive_with_vectors(row_ids, vectors, leaf_capacity, rng)
}

fn partition_recursive_with_vectors(
    row_ids: &[u32],
    vectors: &[Vec<f32>],
    leaf_capacity: usize,
    rng: &mut impl Rng,
) -> PartitionNode {
    let points: Vec<Vec<f32>> = vectors.to_vec();
    let centroid = mean_vector(&points);
    if row_ids.len() <= leaf_capacity {
        let entries: Vec<(u32, Vec<f32>)> = row_ids
            .iter()
            .zip(points.iter())
            .map(|(&id, v)| (id, v.clone()))
            .collect();
        return PartitionNode::Leaf {
            entries,
            centroid,
        };
    }
    let k = row_ids.len().div_ceil(leaf_capacity).clamp(2, 16);
    let (_, assignments) = kmeans_run(&points, k, 20, rng);
    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, &a) in assignments.iter().enumerate() {
        clusters[a].push(row_ids[i]);
    }
    let mut cluster_vectors: Vec<Vec<Vec<f32>>> = vec![Vec::new(); k];
    for (i, &a) in assignments.iter().enumerate() {
        cluster_vectors[a].push(points[i].clone());
    }
    let non_empty_clusters = clusters.iter().filter(|c| !c.is_empty()).count();
    let max_cluster_len = clusters.iter().map(|c| c.len()).max().unwrap_or(0);
    if non_empty_clusters <= 1 || max_cluster_len >= row_ids.len() {
        // Degenerate k-means (e.g. identical points): round-robin by index so each
        // recursive call strictly shrinks the subproblem.
        clusters = vec![Vec::new(); k];
        cluster_vectors = vec![Vec::new(); k];
        for (i, (&id, pt)) in row_ids.iter().zip(points.iter()).enumerate() {
            let a = i % k;
            clusters[a].push(id);
            cluster_vectors[a].push(pt.clone());
        }
    }
    let children: Vec<PartitionNode> = clusters
        .into_iter()
        .zip(cluster_vectors)
        .filter(|(c, _)| !c.is_empty())
        .map(|(c, vecs)| {
            let sub_vectors: Vec<Vec<f32>> = vecs;
            partition_recursive_with_vectors(&c, &sub_vectors, leaf_capacity, rng)
        })
        .collect();
    PartitionNode::Internal { centroid, children }
}

fn mean_vector(points: &[Vec<f32>]) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }
    let dim = points[0].len();
    let mut acc = vec![0.0f32; dim];
    for p in points {
        for (j, &v) in p.iter().enumerate() {
            acc[j] += v;
        }
    }
    let n = points.len() as f32;
    acc.iter().map(|&s| s / n).collect()
}

pub fn max_leaf_size(node: &PartitionNode) -> usize {
    match node {
        PartitionNode::Leaf { entries, .. } => entries.len(),
        PartitionNode::Internal { children, .. } => children.iter().map(max_leaf_size).max().unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_partition_respects_leaf_capacity() {
        let leaf_capacity = 8;
        let vectors: Vec<Vec<f32>> = (0..64)
            .map(|i| vec![i as f32 * 0.1, (i % 7) as f32])
            .collect();
        let row_ids: Vec<u32> = (0..64).collect();
        let tree = PartitionTree::build(&row_ids, &vectors, leaf_capacity, 42);
        assert!(max_leaf_size(&tree.root) <= leaf_capacity);
    }

    #[test]
    fn kmeans_partition_handles_identical_points() {
        let leaf_capacity = 32;
        let n = 1025;
        let duplicate = vec![1.0f32, 2.0, 3.0];
        let vectors: Vec<Vec<f32>> = vec![duplicate.clone(); n];
        let row_ids: Vec<u32> = (0..n as u32).collect();
        let tree = PartitionTree::build(&row_ids, &vectors, leaf_capacity, 7);
        assert!(max_leaf_size(&tree.root) <= leaf_capacity);
        let leaves = tree.all_leaves();
        let total: usize = leaves.iter().map(|(ids, _)| ids.len()).sum();
        assert_eq!(total, n);
    }
}
