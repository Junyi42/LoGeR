// BSD 3-Clause License

// Copyright (c) 2024, Robots Vision and Perception Group

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <tuple>
#include <Eigen/Dense>
#include <iomanip>

// some const
const float MAX_ERROR = 10;
const std::vector<float> PERCENTAGES = {0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55};
const std::vector<std::string> SEQ_TRAIN_NAMES = {
      "colosseo_train0",
      "campus_train0",
      "campus_train1",
      "pincio_train0",
      "spagna_train0",
      "diag_train0",
      "ciampino_train0",
      "ciampino_train1",
      };
const std::vector<std::string> SEQ_TEST_NAMES = {
      "colosseo_test0",
      "campus_test0",
      "campus_test1",
      "pincio_test0",
      "spagna_test0",
      "diag_test0",
      "ciampino_test0",
      "ciampino_test1",
      };

struct Error
{
  int first_frame;
  double r_err, t_err;
  float len;
  Error(int first_frame, double r_err, double t_err, float len) : first_frame(first_frame), r_err(r_err), t_err(t_err), len(len) {}
};

struct ErrorPair
{
  double r_err = 0.;
  double t_err = 0.;
  ErrorPair() = default;
  ErrorPair(double r_err, double t_err) : r_err(r_err), t_err(t_err) {}

  void operator+=(const ErrorPair& e) {
    r_err += e.r_err;
    t_err += e.t_err;
  } 
};

struct Stats
{
  std::string sequence_name_;
  double r_err, t_err;
  Stats(std::string sequence_name, double r_err, double t_err) : sequence_name_(sequence_name), r_err(r_err), t_err(t_err) {}
};

struct Pose
{
  double timestamp;
  Eigen::Isometry3f transform;
  Pose(double timestamp, Eigen::Isometry3f transform) : timestamp(timestamp), transform(transform) {}
};

inline bool sortComparator(const Stats &stat_l, const Stats &stat_r)
{
  return stat_l.t_err < stat_r.t_err;
}

inline std::vector<Pose> loadPoses(const std::string &file_name)
{
  std::vector<Pose> poses;
  std::ifstream file(file_name);
  if (!file.is_open())
  {
    std::cout << "error: unable to open file " << file_name << std::endl;
    return poses;
  }

  while (!file.eof())
  {
    std::string line;
    std::getline(file, line);

    if (line.empty() || line[0] == '#')
      continue;

    std::istringstream iss(line);
    float t, x, y, z, qx, qy, qz, qw;
    if (!(iss >> t >> x >> y >> z >> qx >> qy >> qz >> qw))
    {
      std::cerr << "error reading line from file: " << line << std::endl;
      continue; // Skip this line if unable to read values
    }

    Eigen::Isometry3f P = Eigen::Isometry3f::Identity();
    // float t, x, y, z, qx, qy, qz, qw;
    // file >> t >> x >> y >> z >> qx >> qy >> qz >> qw;

    P.translation() << x, y, z;

    const Eigen::Quaternionf q(qw, qx, qy, qz);
    P.linear() = q.toRotationMatrix();

    poses.push_back(Pose(t, P));
  }

  file.close();
  return poses;
}

inline std::vector<Pose> matchTimestamps(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Pose> poses_matched;
  poses_matched.reserve(poses_gt.size());

  size_t es_index = 0;
  for (const auto &pose_gt : poses_gt)
  {
    // while (es_index < poses_es.size() -  && poses_es[es_index].timestamp < pose_gt.timestamp)
    //   es_index++;

    double min_delta = fabs(poses_es[es_index].timestamp - pose_gt.timestamp);
    while (es_index < poses_es.size() - 1 && fabs(poses_es[es_index + 1].timestamp - pose_gt.timestamp) <= min_delta)
    {
      min_delta = fabs(poses_es[es_index + 1].timestamp - pose_gt.timestamp);
      es_index++;
    }

    poses_matched.push_back(poses_es[es_index]);
  }

  return poses_matched;
}

// Match timestamps for keyframe-only trajectories: use ES timestamps as reference, find closest GT poses
// Returns a pair: (matched_gt_poses, matched_es_poses) with the same length
inline std::pair<std::vector<Pose>, std::vector<Pose>> matchTimestampsKeyframe(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Pose> poses_gt_matched;
  std::vector<Pose> poses_es_matched;
  poses_gt_matched.reserve(poses_es.size());
  poses_es_matched.reserve(poses_es.size());

  constexpr double kMaxTimestampDelta = 1.0;  // Maximum allowed timestamp difference

  size_t gt_index = 0;
  for (const auto &pose_es : poses_es)
  {
    // Find the closest GT pose for this ES pose
    double min_delta = fabs(poses_gt[gt_index].timestamp - pose_es.timestamp);
    while (gt_index < poses_gt.size() - 1 && fabs(poses_gt[gt_index + 1].timestamp - pose_es.timestamp) <= min_delta)
    {
      min_delta = fabs(poses_gt[gt_index + 1].timestamp - pose_es.timestamp);
      gt_index++;
    }

    // Only include if the match is within threshold
    if (min_delta <= kMaxTimestampDelta)
    {
      poses_gt_matched.push_back(poses_gt[gt_index]);
      poses_es_matched.push_back(pose_es);
    }
  }

  return std::make_pair(poses_gt_matched, poses_es_matched);
}

inline std::vector<float> trajectoryDistances(const std::vector<Pose> &poses)
{
  std::vector<float> dist;
  dist.push_back(0);
  for (size_t i = 1; i < poses.size(); ++i)
  {
    const Eigen::Vector3f t1 = poses[i - 1].transform.translation();
    const Eigen::Vector3f t2 = poses[i].transform.translation();

    dist.push_back(dist[i - 1] + (t1 - t2).norm());
  }
  return dist;
}

inline size_t lastFrameFromSegmentLength(const std::vector<float> &dist, const int &first_frame, const float &len)
{
  for (size_t i = first_frame; i < dist.size(); ++i)
    if (dist[i] > dist[first_frame] + len)
      return i;
  return -1;
}

inline double rotationError(const Eigen::Isometry3f &pose_error)
{
  Eigen::Quaternionf q(pose_error.linear());
  q.normalize();

  const Eigen::Quaternionf q_identity(1.0f, 0.0f, 0.0f, 0.0f);
  const double error_radians = q_identity.angularDistance(q);

  const double error_degrees = error_radians * (180.0f / M_PI);
  return error_degrees;
}

inline double translationError(const Eigen::Isometry3f &pose_error)
{
  const Eigen::Vector3f t = pose_error.translation();
  return t.norm();
}

inline std::vector<Error> computeSequenceErrors(const std::vector<Pose> poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Error> err;

  const std::vector<float> dist = trajectoryDistances(poses_gt);
  const float seq_length = dist.back();
  std::cout << "sequence length [m]: " << seq_length << std::endl;

  std::vector<float> lengths;
  for (const float &percentage : PERCENTAGES)
  {
    const float len = seq_length * percentage;
    lengths.push_back(len);
    std::cout << "percentage: " << percentage << ", subsequence length [m]: " << len << std::endl;
  }
  std::cout << std::endl;

  for (size_t first_frame = 0; first_frame < poses_gt.size(); ++first_frame)
  {
    for (size_t i = 0; i < lengths.size(); ++i)
    {

      const float curr_len = lengths[i];
      const int last_frame = lastFrameFromSegmentLength(dist, first_frame, curr_len);

      if (last_frame == -1)
        continue;

      const Eigen::Isometry3f pose_delta_gt = poses_gt[first_frame].transform.inverse() * poses_gt[last_frame].transform;
      const Eigen::Isometry3f pose_delta_es = poses_es[first_frame].transform.inverse() * poses_es[last_frame].transform;
      const Eigen::Isometry3f pose_error = pose_delta_es.inverse() * pose_delta_gt;
      const double r_err = rotationError(pose_error);
      const double t_err = translationError(pose_error);

      err.push_back(Error(first_frame, r_err / curr_len, t_err / curr_len, curr_len));
    }
  }

  return err;
}

inline std::vector<Pose> computeAlignedEstimate(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es)
{
  std::vector<Pose> poses_es_aligned;
  poses_es_aligned.reserve(poses_es.size());

  Eigen::Matrix<float, 3, Eigen::Dynamic> gt_matrix;
  gt_matrix.resize(Eigen::NoChange, poses_gt.size());
  for (size_t i = 0; i < poses_gt.size(); ++i)
    gt_matrix.col(i) = poses_gt[i].transform.translation();

  Eigen::Matrix<float, 3, Eigen::Dynamic> es_matrix;
  es_matrix.resize(Eigen::NoChange, poses_es.size());
  for (size_t i = 0; i < poses_es.size(); ++i)
    es_matrix.col(i) = poses_es[i].transform.translation();

  // last argument to true for monocular (Sim3)
  const Eigen::Matrix4f transform_matrix = Eigen::umeyama(es_matrix, gt_matrix, true);
  Eigen::Isometry3f transform = Eigen::Isometry3f(transform_matrix.block<3, 3>(0, 0));
  transform.translation() = transform_matrix.block<3, 1>(0, 3);

  for (size_t i = 0; i < poses_es.size(); ++i)
    poses_es_aligned.push_back(Pose(poses_es[i].timestamp, transform * poses_es[i].transform));

  return poses_es_aligned;
}

inline std::pair<Stats, std::vector<ErrorPair>> computeSequenceRPE(const std::vector<Error> &seq_err, const std::string &sequence_name, size_t num_poses)
{
  double t_err = 0;
  double r_err = 0;

  std::vector<ErrorPair> RPE_errors;
  RPE_errors.resize(num_poses);
  std::vector<int> count;
  count.resize(num_poses);

  for (const Error &error : seq_err)
  {
    RPE_errors[error.first_frame] += ErrorPair(error.r_err, error.t_err);
    count[error.first_frame]++;
    t_err += error.t_err;
    r_err += error.r_err;
  }

  for (size_t i = 0; i < num_poses; ++i) {
    if(!count[i])
      continue;
    RPE_errors[i].r_err /= count[i];
    RPE_errors[i].t_err /= count[i];
  }

  const double r_rpe = r_err / double(seq_err.size());
  const double t_rpe = 100 * t_err / double(seq_err.size());
  return std::make_pair(Stats(sequence_name, r_rpe, t_rpe), RPE_errors);
}

inline std::tuple<Stats, Stats, std::vector<ErrorPair>> computeSequenceATE(const std::vector<Pose> &poses_gt, const std::vector<Pose> &poses_es_aligned, const std::string &sequence_name)
{
  double r_sq_sum = 0;
  double t_sq_sum = 0;
  double r_sum = 0;
  double t_sum = 0;
  std::vector<ErrorPair> ATE_errors;
  ATE_errors.reserve(poses_gt.size());

  for (size_t i = 0; i < poses_gt.size(); ++i)
  {
    const Eigen::Isometry3f pose_error = poses_gt[i].transform.inverse() * poses_es_aligned[i].transform;
    const double r_err = rotationError(pose_error);
    const double t_err = translationError(pose_error);

    ATE_errors.push_back(ErrorPair(r_err, t_err));

    r_sq_sum += r_err * r_err;
    t_sq_sum += t_err * t_err;
    r_sum += r_err;
    t_sum += t_err;
  }

  const double n = double(poses_gt.size());
  const double r_ate_rmse = std::sqrt(r_sq_sum / n);
  const double t_ate_rmse = std::sqrt(t_sq_sum / n);
  // legacy metric: sqrt of mean of (unsquared) errors, kept for backward comparison
  const double r_ate_legacy = std::sqrt(r_sum / n);
  const double t_ate_legacy = std::sqrt(t_sum / n);
  return std::make_tuple(Stats(sequence_name, r_ate_rmse, t_ate_rmse),
                         Stats(sequence_name, r_ate_legacy, t_ate_legacy),
                         ATE_errors);
}

inline void computeRank(std::vector<Stats> &stats, const std::string &path_to_result_file, const std::string &path_to_rank_file)
{
  // sort(stats.begin(), stats.end(), sortComparator);

  FILE *fp = fopen(path_to_result_file.c_str(), "w");

  double rank = 0;
  for (const Stats stat : stats)
  {
    fprintf(fp, "%f %f\n", stat.t_err, stat.r_err);
    std::cout << stat.sequence_name_ << " " << stat.t_err << " " << stat.r_err;

    if (stat.t_err > MAX_ERROR)
    {
      std::cout << " - exceeded max error" << std::endl;
      continue;
    }

    rank += (double)(MAX_ERROR - stat.t_err);
    std::cout << std::endl;
  }

  if (!stats.empty()) {
    double t_err_sum = 0;
    double r_err_sum = 0;
    for (const auto& stat : stats) {
      t_err_sum += stat.t_err;
      r_err_sum += stat.r_err;
    }
    double t_err_avg = t_err_sum / stats.size();
    double r_err_avg = r_err_sum / stats.size();
    
    fprintf(fp, "Average: %f %f\n", t_err_avg, r_err_avg);
    std::cout << "Average: " << t_err_avg << " " << r_err_avg << std::endl;
  }

  fclose(fp);

  fp = fopen(path_to_rank_file.c_str(), "w");
  fprintf(fp, "%f\n", rank);
  fclose(fp);

  std::cout << "rank: " << rank << std::endl
            << std::endl;
}

// Copyright 2019 ETH Zürich, Thomas Schöps
void WriteTrajectorySVG(
    std::ofstream& stream,
    int plot_size_in_pixels,
    const Eigen::Vector3f& min_vec,
    const Eigen::Vector3f& max_vec,
    const std::vector<Pose>& trajectory,
    const std::string& color,
    const float stroke_width,
    int dimension1,
    int dimension2) {
  constexpr double kTimestampHoleThreshold = 0.07;
  
  std::ostringstream stroke_width_stream;
  stroke_width_stream << stroke_width;
  std::string stroke_width_string = stroke_width_stream.str();
  
  std::ostringstream half_stroke_width_stream;
  half_stroke_width_stream << (0.5 * stroke_width);
  std::string half_stroke_width_string = half_stroke_width_stream.str();
  
  bool within_polyline = false;
  
  for (std::size_t i = 0; i < trajectory.size() - 1; ++ i) {
    const Eigen::Vector3f& point = trajectory[i].transform.translation();
    Eigen::Vector3f plot_point = plot_size_in_pixels * (point - min_vec).cwiseQuotient(max_vec - min_vec);
    
    // Is the segment [i, i + 1] valid (or is it a hole)?
    bool segment_valid = (trajectory[i+1].timestamp - trajectory[i].timestamp <= kTimestampHoleThreshold);
    
    if (!segment_valid && !within_polyline) {
      stream << "<circle cx=\"" << plot_point.coeff(dimension1) << "\" cy=\"" << plot_point.coeff(dimension2) << "\" r=\"" << half_stroke_width_string << "\" fill=\"" << color << "\"/>\n";
      continue;
    }
    
    if (segment_valid && !within_polyline) {
      // Start new polyline
      stream << "<polyline points=\"";
      within_polyline = true;
    } else {
      // Write the space between two points
      stream << " ";
    }
    
    stream << plot_point.coeff(dimension1) << "," << plot_point.coeff(dimension2);
    
    if (!segment_valid && within_polyline) {
      // End polyline
      stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
      within_polyline = false;
    }
  }
  
  if (within_polyline) {
    // End polyline
    stream << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width_string << "\" fill=\"none\" />\n";
    // within_polyline = false;
  }
}

// Copyright 2019 ETH Zürich, Thomas Schöps
void PlotTrajectories(
    const std::string& path,
    int plot_size_in_pixels,
    const std::vector<Pose>& ground_truth,
    const std::vector<Pose>& aligned_estimate,
    int dimension1,
    int dimension2) {
  std::ofstream stream(path, std::ios::out);
  
  stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  stream << "<svg width=\"" << plot_size_in_pixels << "\" height=\"" << plot_size_in_pixels
                          << "\" viewBox=\"0 0 " << plot_size_in_pixels << " " << plot_size_in_pixels
                          << "\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n";
  
  // Determine plot extent based on the ground truth trajectory
  Eigen::Vector3f min_vec = Eigen::Vector3f::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3f max_vec = Eigen::Vector3f::Constant(-1 * std::numeric_limits<double>::infinity());

  for (std::size_t i = 0; i < ground_truth.size(); ++ i) {
    min_vec = min_vec.cwiseMin(ground_truth[i].transform.translation());
    max_vec = max_vec.cwiseMax(ground_truth[i].transform.translation());
  }

  for (std::size_t i = 0; i < aligned_estimate.size(); ++ i) {
    min_vec = min_vec.cwiseMin(aligned_estimate[i].transform.translation());
    max_vec = max_vec.cwiseMax(aligned_estimate[i].transform.translation());
  }
  
  float largest_size = (max_vec - min_vec).maxCoeff();
  Eigen::Vector3f center = 0.5 * (min_vec + max_vec);
  constexpr float kSizeExtensionFactor = 1.1f;
  min_vec = center - 0.5 * Eigen::Vector3f::Constant(kSizeExtensionFactor * largest_size);
  max_vec = center + 0.5 * Eigen::Vector3f::Constant(kSizeExtensionFactor * largest_size);
  
  // Plot ground truth trajectory
  WriteTrajectorySVG(stream, plot_size_in_pixels, min_vec, max_vec, ground_truth, "green", 1, dimension1, dimension2);
  
  // Plot estimated trajectory
  WriteTrajectorySVG(stream, plot_size_in_pixels, min_vec, max_vec, aligned_estimate, "red", 1, dimension1, dimension2);
  
  stream << "</svg>\n";
  
  stream.close();
}

inline void dumpError(const std::string& path, const std::vector<ErrorPair>& errors, const std::vector<Pose>& poses_es, bool rpe=false) {
  std::ofstream file(path);

  if(rpe)
    file << "# ts rotation(%) translation(%)" << std::endl;
  else
    file << "# ts rotation(deg) translation(m)" << std::endl;

  for (size_t i = 0; i < poses_es.size(); ++i) {
    file << poses_es[i].timestamp << " " << errors[i].r_err << " " << errors[i].t_err << std::endl;
  }

  file.close();
}

inline void eval(const std::string &path_to_gt, const std::string &path_to_es, const std::string &eval_type, bool plot, int max_frames = -1, bool align_kf = false)
{
  std::string path_to_result;
  if (max_frames > 0) {
    path_to_result = path_to_es + "/results_" + std::to_string(max_frames) + "/" + eval_type;
  } else {
    path_to_result = path_to_es + "/results/" + eval_type;
  }
  if (align_kf) {
    path_to_result += "_kf";
  }
  system(("mkdir -p " + path_to_result).c_str());

  std::vector<std::string> seq_names;
  if (eval_type == "train")
    seq_names = SEQ_TRAIN_NAMES;
  else if (eval_type == "test")
    seq_names = SEQ_TEST_NAMES;
  else
    return;

  std::vector<Stats> rpe_stats;
  std::vector<Stats> ate_stats;
  std::vector<Stats> ate_legacy_stats;
  for (size_t i = 0; i < seq_names.size(); ++i)
  {
    const std::string sequence_name = seq_names[i];
    const std::string path_to_gt_file = path_to_gt + "/" + sequence_name + "_gt.txt";
    const std::string path_to_es_file = path_to_es + "/" + sequence_name + "_es.txt";

    std::vector<Pose> poses_gt_orig = loadPoses(path_to_gt_file);
    std::vector<Pose> poses_es_unmatched = loadPoses(path_to_es_file);

    if (max_frames > 0) {
      if (poses_gt_orig.size() > max_frames) poses_gt_orig.erase(poses_gt_orig.begin() + max_frames, poses_gt_orig.end());
      if (poses_es_unmatched.size() > max_frames) poses_es_unmatched.erase(poses_es_unmatched.begin() + max_frames, poses_es_unmatched.end());
    }

    std::cout << "=============================================" << std::endl;
    std::cout << "processing: " << sequence_name << std::endl;
    std::cout << "estimated poses: " << poses_es_unmatched.size() << std::endl;
    std::cout << "gt poses: " << poses_gt_orig.size() << std::endl;

    if (poses_gt_orig.size() == 0 || poses_es_unmatched.size() == 0)
    {
      std::cout << "ERROR: could not read (all) poses of: " << sequence_name << std::endl;
      continue;
    }

    std::vector<Pose> poses_gt;
    std::vector<Pose> poses_es;
    
    if (align_kf) {
      // Keyframe mode: match based on ES timestamps, only keep matched pairs
      const auto [gt_matched, es_matched] = matchTimestampsKeyframe(poses_gt_orig, poses_es_unmatched);
      poses_gt = gt_matched;
      poses_es = es_matched;
      std::cout << "keyframe mode: matched " << poses_es.size() << " pose pairs" << std::endl;
    } else {
      // Standard mode: match based on GT timestamps
      poses_gt = poses_gt_orig;
      poses_es = matchTimestamps(poses_gt, poses_es_unmatched);
    }
    
    std::cout << "matched poses: " << poses_es.size() << std::endl
              << std::endl;

    if (poses_gt.size() == 0 || poses_es.size() == 0)
    {
      std::cout << "ERROR: no valid matched poses for: " << sequence_name << std::endl;
      continue;
    }

    const std::vector<Error> seq_err = computeSequenceErrors(poses_gt, poses_es);
    const auto [rpe_stat, RPE_errors] = computeSequenceRPE(seq_err, sequence_name, poses_es.size());
    rpe_stats.push_back(rpe_stat);

    const std::vector<Pose> poses_es_aligned = computeAlignedEstimate(poses_gt, poses_es);
    const auto [ate_stat, ate_legacy_stat, ATE_errors] = computeSequenceATE(poses_gt, poses_es_aligned, sequence_name);
    ate_stats.push_back(ate_stat);
    ate_legacy_stats.push_back(ate_legacy_stat);

    if (plot) {
      constexpr int kPlotSize = 600;  // in pixels; this is the default display size of the SVGs
      PlotTrajectories(
          path_to_result + "/" + sequence_name + "_top.svg",
          kPlotSize,
          poses_gt,
          poses_es_aligned,
          0,
          1);
      PlotTrajectories(
          path_to_result + "/" + sequence_name + "_front.svg",
          kPlotSize,
          poses_gt,
          poses_es_aligned,
          0,
          2);
      PlotTrajectories(
          path_to_result + "/" + sequence_name + "_side.svg",
          kPlotSize,
          poses_gt,
          poses_es_aligned,
          1,
          2);
      dumpError(path_to_result + "/" + sequence_name + "_ate.txt", ATE_errors, poses_es);
      dumpError(path_to_result + "/" + sequence_name + "_rpe.txt", RPE_errors, poses_es, true);
    }
  }

  std::cout << eval_type << " stats RPE (sequence, t_err [%], r_err [deg/m]):" << std::endl;
  computeRank(rpe_stats, path_to_result + "/results_rpe.txt", path_to_result + "/rank_rpe.txt");

  std::cout << eval_type << " stats ATE RMSE (sequence, t_err [m], r_err [deg]):" << std::endl;
  computeRank(ate_stats, path_to_result + "/results_ate.txt", path_to_result + "/rank_ate.txt");

  std::cout << eval_type << " stats ATE legacy (sequence, t_err [sqrt(m)], r_err [sqrt(deg)]):" << std::endl;
  computeRank(ate_legacy_stats, path_to_result + "/results_ate_legacy.txt", path_to_result + "/rank_ate_legacy.txt");
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cout << "usage: ./vbr_benchmark path_to_gt path_to_es [--plot] [--align_kf]" << std::endl;
    std::cout << "  --plot     : generate SVG trajectory plots" << std::endl;
    std::cout << "  --align_kf : keyframe mode for sparse/partial trajectories" << std::endl;
    return 1;
  }

  const std::string &path_to_gt = argv[1];
  const std::string &path_to_es = argv[2];

  bool plot = false;
  bool align_kf = false;
  
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--plot") {
      plot = true;
    } else if (arg == "--align_kf") {
      align_kf = true;
    }
  }

  if (align_kf) {
    std::cout << "Keyframe mode enabled: using ES timestamps as reference for matching" << std::endl;
  }

  for (int n = 1000; n <= 19000; n += 2000) {
    std::cout << "Evaluating with max frames: " << n << std::endl;
    eval(path_to_gt, path_to_es, "train", plot, n, align_kf);
    eval(path_to_gt, path_to_es, "test", plot, n, align_kf);
  }

  std::cout << "If you are running this at home, you do not have test set ground truth, so ignore error messages related to test!" << std::endl;

  return 0;
}