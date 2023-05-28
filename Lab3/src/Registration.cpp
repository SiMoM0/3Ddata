#include "Registration.h"

#include <utility>


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  PointDistance(Eigen::Vector3d  source, Eigen::Vector3d  target)
    : source(source), target(target) {}

  template <typename T>
  bool operator()(const T* const transformation, T* residual) const {
      // extract source data
      Eigen::Matrix<T, 3, 1> source_point;
      source_point << T(source[0]), T(source[1]), T(source[2]);
      // Extract the translation parameters
      T tx = transformation[3];
      T ty = transformation[4];
      T tz = transformation[5];

      // Apply the rotation and translation to the source point
      Eigen::Matrix<T, 3, 1> rotated_point;
      ceres::AngleAxisRotatePoint(transformation, source_point.data(), rotated_point.data());
      rotated_point[0] += tx;
      rotated_point[1] += ty;
      rotated_point[2] += tz;

      // Compute the residual as the difference between the transformed source point and the target point
      residual[0] = rotated_point.x() - target.x();
      residual[1] = rotated_point.y() - target.y();
      residual[2] = rotated_point.z() - target.z();

      return true;
  }

  static ceres::CostFunction* Create(Eigen::Vector3d& source, Eigen::Vector3d& target) {
      return new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(new PointDistance(source, target));
  }

  Eigen::Vector3d source;
  Eigen::Vector3d target;
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // rmse of previous iteration
  double prev_rmse = 0.0;

  for(int i=0; i<max_iteration; ++i) {
      std::printf("Iteration [%d]\n", i);
      std::tuple<std::vector<size_t>, std::vector<size_t>, double> tuple = find_closest_point(threshold);

      // extract rmse of current iteration
      double rmse = std::get<2>(tuple);

      // check convergence
      if(std::abs(rmse - prev_rmse) < relative_rmse)
          break;

      // update
      prev_rmse = rmse;

      Eigen::Matrix4d prev_transformation = get_transformation();
      Eigen::Matrix4d curr_transformation, new_transformation = Eigen::Matrix4d::Identity();

      if (mode == "svd") {
          curr_transformation = get_svd_icp_transformation(std::get<0>(tuple), std::get<1>(tuple));
      } else if (mode == "lm") {
          curr_transformation = get_lm_icp_registration(std::get<0>(tuple), std::get<1>(tuple));
      }

      new_transformation.block<3, 3>(0, 0) = curr_transformation.block<3, 3>(0, 0) * prev_transformation.block<3, 3>(0, 0);
      new_transformation.block<3, 1>(0, 3) = curr_transformation.block<3, 3>(0, 0) * prev_transformation.block<3, 1>(0, 3) + curr_transformation.block<3, 1>(0, 3);

      set_transformation(new_transformation);

      std::cout << curr_transformation << std::endl;

      source_for_icp_.Transform(curr_transformation);
  }

  return;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse;

  // compute KD Tree for target point cloud
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  std::vector<int> idx(1);
  std::vector<double> dist2(1);

  // clone source point cloud
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  // number of points in source point cloud
  int num_points = source_clone.points_.size();

  // find closest for each point in source point cloud
  for(size_t i=0; i<num_points; ++i) {
      source_point = source_clone.points_[i];
      target_kd_tree.SearchKNN(source_point, 1, idx, dist2);

      // check if distance is lower than the threshold
      if(sqrt(dist2[0]) <= threshold) {
          source_indices.push_back(i);
          target_indices.push_back(idx[0]);
          rmse = rmse * i/(i+1) + dist2[0]/(i+1);
      }
  }

  rmse = sqrt(rmse);

  // std::printf("RMSE in find_closest_point = %f\n", rmse);

  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  // clone source point cloud
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  // initialize centroids for both point clouds
  Eigen::Vector3d source_centroid = std::get<0>(source_clone.ComputeMeanAndCovariance());
  Eigen::Vector3d target_centroid = std::get<0>(target_.ComputeMeanAndCovariance());

  Eigen::Matrix3d W;

  // compute matrix W
  for(int i=0; i<source_indices.size(); ++i) {
      // get normalized points
      Eigen::Vector3d d = source_clone.points_[source_indices[i]] - source_centroid;
      Eigen::Vector3d m = target_.points_[target_indices[i]] - target_centroid;

      // update W
      W += m * d.transpose();
  }

  // perform SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Compute the rotation matrix
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();

  // Handle the special reflection case
  if (svd.matrixU().determinant() * svd.matrixV().determinant() == -1) {
      // define diagonal matrix with (1, 1, -1)
      Eigen::DiagonalMatrix<double, 3> diagonal_matrix;
      diagonal_matrix.diagonal() << 1.0, 1.0, -1.0;
      // Update R
      R = svd.matrixU() * diagonal_matrix * svd.matrixV().transpose();
  }

  // compute translation vector
  Eigen::Vector3d t = target_centroid - R * source_centroid;

  // std::cout << R << " " << t << std::endl;

  // set transformation matrix with R and t
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Problem problem;
  ceres::Solver::Summary summary;
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  // clone source point cloud
  open3d::geometry::PointCloud source_clone = source_for_icp_;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();

  Eigen::Vector3d source_point, target_point;

  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    source_point = source_clone.points_[source_indices[i]];
    target_point = target_.points_[target_indices[i]];

    ceres::CostFunction* cost_function = PointDistance::Create(source_point, target_point);
    problem.AddResidualBlock(cost_function, NULL, transformation_arr.data());
  }

  // solve problem
  Solve(options, &problem, &summary);

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


