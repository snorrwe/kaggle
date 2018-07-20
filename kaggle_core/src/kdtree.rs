#[macro_export]
macro_rules! make_kd_tree {
    ($k:expr, $value_type:ty, $name:ident) => {
        mod $name {
            use super::*;
            use std::cmp::Ordering;
            use std::rc::Rc;

            pub const SIZE: usize = $k;
            pub type Vector = [f32; SIZE];
            pub type TValue = $value_type;
            pub type InitPoints = (Vector, Rc<TValue>);
            pub type QueryResult = (f32, Vector, Rc<TValue>);

            #[derive(Debug, Clone)]
            pub struct Node {
                left: Option<Box<Node>>,
                right: Option<Box<Node>>,
                value: Rc<TValue>,
                position: Vector,
            }

            pub fn new(mut nodes: &mut [InitPoints]) -> Option<Box<Node>> {
                from_depth(&mut nodes, 0)
            }

            pub fn from_depth(nodes: &mut [InitPoints], depth: usize) -> Option<Box<Node>> {
                if nodes.len() == 0 {
                    return None;
                }

                let axis = depth % SIZE;

                nodes.sort_unstable_by(|x, y| {
                    let x = x.0[axis];
                    let y = y.0[axis];
                    if x < y {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                });

                let median = nodes.len() / 2;

                let left = from_depth(&mut nodes[0..median], depth + 1);
                let right = from_depth(&mut nodes[median + 1..], depth + 1);

                let node = &nodes[median];
                let result = Node {
                    left: left,
                    right: right,
                    position: node.0,
                    value: node.1.clone(),
                };
                Some(Box::new(result))
            }

            fn euclidean(x: &Vector, y: &Vector) -> f32 {
                let mut sum = 0.0;
                for i in 0..SIZE {
                    sum += (x[i] - y[i]).powi(2);
                }
                sum.sqrt()
            }

            impl Node {
                pub fn get_coord(&self) -> &Vector {
                    &self.position
                }

                pub fn clone_coord(&self) -> Vector {
                    self.get_coord().clone()
                }

                pub fn get_value(&self) -> Rc<TValue> {
                    self.value.clone()
                }

                pub fn find_k_nearest(&self, vector: Vector, k: usize) -> Vec<QueryResult> {
                    self.find_k_nearest_by(&vector, k, euclidean)
                }

                pub fn find_k_nearest_by<TDistance>(
                    &self,
                    vector: &Vector,
                    k: usize,
                    distance: TDistance,
                ) -> Vec<QueryResult>
                where
                    TDistance: Fn(&Vector, &Vector) -> f32,
                {
                    self.find_k_nearest_by_depth(vector, k, 0, &distance)
                        .iter()
                        .map(|x| {
                            let node = x.1;
                            (x.0, node.clone_coord(), node.get_value())
                        })
                        .collect()
                }

                fn find_k_nearest_by_depth<'a, TDistance>(
                    &'a self,
                    vector: &Vector,
                    k: usize,
                    depth: usize,
                    distance: &TDistance,
                ) -> Vec<(f32, &'a Node)>
                where
                    TDistance: Fn(&Vector, &Vector) -> f32,
                {
                    if self.left.is_none() && self.right.is_none() {
                        let d = distance(&self.position, vector);
                        return vec![(d, self)];
                    }
                    let axis = depth % SIZE;
                    let (nearer, further) = {
                        if self.right.is_none()
                            || (self.left.is_some() && vector[axis] <= self.position[axis])
                        {
                            (self.left.as_ref().unwrap(), &self.right)
                        } else {
                            (self.right.as_ref().unwrap(), &self.left)
                        }
                    };
                    let mut result = nearer.find_k_nearest_by_depth(vector, k, depth + 1, distance);

                    if let Some(ref further) = further {
                        if result.len() < k
                            || distance(&further.position, vector) < result.last().unwrap().0
                        {
                            result.append(&mut further.find_k_nearest_by_depth(
                                vector,
                                k,
                                depth + 1,
                                distance,
                            ));
                        }
                    }

                    result.push((distance(&self.position, vector), self));

                    result.sort_unstable_by(|x, y| {
                        if x.0 < y.0 {
                            Ordering::Less
                        } else {
                            Ordering::Greater
                        }
                    });

                    result.truncate(k);
                    result
                }
            }

        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn can_create_kdtree() {
        make_kd_tree!(2, u8, test_kd_tree);
        test_kd_tree::new(&mut vec![
            ([1.0, 2.0], Rc::new(55)),
            ([2.0, 2.0], Rc::new(55)),
            ([5.0, 2.0], Rc::new(55)),
            ([1.0, 8.0], Rc::new(55)),
            ([1.0, 2.0], Rc::new(55)),
            ([9.0, 2.5], Rc::new(55)),
            ([1.0, 2.1], Rc::new(55)),
            ([1.3, 2.8], Rc::new(55)),
        ]).unwrap();
    }

    #[test]
    fn can_find_k_nearest_easy() {
        make_kd_tree!(2, u8, test_kd_tree);
        let tree = test_kd_tree::new(&mut vec![
            ([0.0, 5.0], Rc::new(0)),
            ([5.0, 0.0], Rc::new(0)),
            ([5.0, 2.0], Rc::new(0)),
            ([1.0, 8.0], Rc::new(3)),
            ([1.0, 20.0], Rc::new(4)),
            ([90.0, 2.5], Rc::new(5)),
            ([10.0, 2.1], Rc::new(6)),
            ([1.3, 25.8], Rc::new(7)),
        ]).unwrap();

        let result = tree.find_k_nearest([0.0, 0.0], 3);
        assert_eq!(result.len(), 3);
        result.iter().for_each(|x| assert_eq!(*x.2, 0));
    }

    #[test]
    fn can_find_k_nearest_easy_shuffled() {
        make_kd_tree!(2, u8, test_kd_tree);
        let tree = test_kd_tree::new(&mut vec![
            ([1.3, 25.8], Rc::new(7)),
            ([5.0, 0.0], Rc::new(0)),
            ([1.0, 8.0], Rc::new(3)),
            ([10.0, 2.1], Rc::new(6)),
            ([1.0, 20.0], Rc::new(4)),
            ([0.0, 5.0], Rc::new(0)),
            ([90.0, 2.5], Rc::new(5)),
            ([5.0, 2.0], Rc::new(0)),
        ]).unwrap();

        let result = tree.find_k_nearest([0.0, 0.0], 3);
        println!("{:#?}", result);
        assert_eq!(result.len(), 3);
        result.iter().for_each(|x| assert_eq!(*x.2, 0));

        let query_pos = [100.0, 100.0];
        let result = tree.find_k_nearest(query_pos.clone(), 3);
        assert_eq!(result.len(), 3);
        result.iter().for_each(|x| assert!(*x.2 != 0));
    }
}
