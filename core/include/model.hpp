#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"
#include "pose_refinement_state.hpp"
#include "camera_pose_conditioning.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

int numShBases(int degree);
/// Validate checkpoint structure and tensor metadata without allocating Metal
/// buffers or changing model state. Throws std::runtime_error when invalid.
void validateCheckpointFile(const std::string &filename);

struct ModelPoseRefinementState {
  bool anchor = false;
  uint32_t optimizerStepCount = 0;
  msplat::detail::PoseRefinementGeometry geometry;
  const char* frameId = nullptr;
  size_t frameIdLength = 0;
};

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const float* bgColor = nullptr,
        int stopDensifyAt = -1,
        int maxGaussians = -1,
        bool refinePhotometricGains = false,
        bool refineCameraPoses = false,
        int poseAnchorCameraIndex = -1,
        msplat::CameraPoseConditioning cameraPoseConditioning =
            msplat::CameraPoseConditioning::Raw,
        bool transparentTrainingMasks = false,
        float transparentAlphaLossWeight = 0.1f);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  void saveSpz(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step);
  // Compatibility path for callers that do not load training masks.
  void fullIteration(Camera& cam, int step, MTensor& gt, float ssimWeight);
  void fullIteration(Camera& cam, size_t cameraIndex, int step,
                     MTensor& gt, float ssimWeight);
  void fullIteration(Camera& cam, int step,
                     const CameraTrainingTarget& target, float ssimWeight);
  void fullIteration(Camera& cam, size_t cameraIndex, int step,
                     const CameraTrainingTarget& target, float ssimWeight);
  MTensor render(Camera& cam, int step);

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  /// Bytes held by the model's own GPU buffers — parameters, Adam state,
  /// and the densification scratch. Sized by capacity, not active count.
  size_t estimatedGpuBytes() const;
  uint32_t poseRefinementStateCount() const;
  /// The caller must synchronize Metal before reading the returned tensor
  /// values. The borrowed frame ID remains valid for this Model's lifetime.
  ModelPoseRefinementState poseRefinementState(
      uint32_t canonicalCameraIndex) const;
  void allocateDensificationScratch();
  void resetDensificationScratch();
  void retireDensificationState();
  bool hasDensificationScratch() const;
  void ensureCapacity(int needed);
  int capacityFor(int needed) const;

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;
  MTensor densify_random_samples;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;

  MTensor backgroundColor;

  // Per-canonical-camera log-domain RGB gain. Applying exp(logGain) to the
  // render inside the loss models source exposure/white-balance variation
  // without changing canonical renders or exported Gaussian colors.
  MTensor cameraLogGains;
  MTensor cameraLogGainExpAvg;
  MTensor cameraLogGainExpAvgSq;
  std::vector<uint32_t> cameraLogGainStepCounts;
  // Optional geometric camera refinement. Rows are indexed by the canonical
  // dataset camera index. Each delta is [camera-space translation, axis-angle]
  // and left-multiplies the immutable renderer view matrix during training.
  MTensor cameraPoseDeltas;
  MTensor cameraPoseExpAvg;
  MTensor cameraPoseExpAvgSq;
  MTensor cameraPosePreconditioners;
  std::vector<uint32_t> cameraPoseStepCounts;
  std::vector<std::string> cameraFrameIds;
  std::vector<float> cameraBasePoses;
  std::vector<float> cameraPosePreconditionerValues;
  std::vector<uint8_t> cameraPosePreconditionerReady;
  std::vector<float> cameraPosePointPool;
  std::vector<uint64_t> cameraPosePointIds;

  int numCameras;
  int datasetCameraCount;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  /// Checkpoint SH degree required by this model's fixed parameter layout.
  int configuredSHDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  /// Hard population and backing-buffer limit. -1 means unlimited.
  int maxGaussians;
  bool refinePhotometricGains;
  bool refineCameraPoses;
  int poseAnchorCameraIndex;
  msplat::CameraPoseConditioning cameraPoseConditioning;
  bool transparentTrainingMasks;
  float transparentAlphaLossWeight;
  bool keepCrs;

  float scale;
  float translation[3] = {};

private:
  void ensureCameraPosePreconditioner(
      const Camera& camera, size_t canonicalCameraIndex);
};

#endif
