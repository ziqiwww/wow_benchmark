#include "dataloader.hh"
#include "micro.hh"
#include "type.hh"
#include "apps/gtmanager.hh"

namespace spatt {

class ResultAnalyser
{
public:
  ResultAnalyser(const std::vector<std::vector<label_t>> &gt) : gt_(gt), nq_{0} {}

  ~ResultAnalyser() = default;

  DISABLE_COPY_MOVE_AND_ASSIGN(ResultAnalyser);

  void LoadGT(const std::string &gt_file);

  // calculate recall@k of query i
  auto CalculateRecall(label_t q_i, size_t k, const std::vector<dist_id_pair> &result) -> double;

  void Step(double time, double distance, double hop, double recall)
  {
    tot_time_ += time;
    tot_recall_ += recall;
    tot_distance_ += distance;
    tot_hop_ += hop;
    nq_++;
  }

  void Finalize()
  {
    if(nq_ == 0) {
      LOG("No query");
      return;
    }
    QPS_ = nq_ / tot_time_;
    tot_recall_ /= nq_;
    tot_distance_ /= nq_;
    tot_hop_ /= nq_;
  }

  void Dump(const std::string &output, SearchParameters sp)
  {
    // add appendly
    std::ofstream out(output, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
      LOG(fmt::format("Failed to open {}", output));
      std::abort();
    }
    // write a csv keep 6 decimal
    out << fmt::format("{},{:.6f},{:.6f},{:.6f},{:.6f},{},{}\n",
        sp.efs_,
        tot_recall_,
        QPS_,
        tot_distance_,
        tot_hop_,
        sp.layer_rng_.l_,
        sp.layer_rng_.u_);
    LOG(fmt::format("efs: {}, total time: {}, Average recall: {}, QPS: {} Average distance: {} Average hop: {}, Lowest "
                    "layer: {}, Highest layer: {}",
        sp.efs_,
        tot_time_,
        tot_recall_,
        QPS_,
        tot_distance_,
        tot_hop_,
        sp.layer_rng_.l_,
        sp.layer_rng_.u_));
    out.close();
  }

private:
  double                                   tot_time_{};
  double                                   tot_recall_{};
  double                                   tot_distance_{};
  double                                   tot_hop_{};
  double                                   QPS_{};
  const std::vector<std::vector<label_t>> &gt_;
  int                                     nq_;
};

inline auto ResultAnalyser::CalculateRecall(label_t q_i, size_t k, const std::vector<dist_id_pair> &result) -> double
{
  if (gt_.empty()) {
    LOG("Ground truth is empty");
    return 0.0;
  }
  if (q_i >= gt_.size()) {
    LOG(fmt::format("Query index {} out of range", q_i));
    std::abort();
  }
  const auto &gt = gt_[q_i];
  if (k > gt.size()) {
    k = gt.size();
  }
  size_t n = 0;
  for (const auto &[d, id] : result) {
    if (std::find(gt.begin(), gt.end(), id) != gt.end()) {
      n++;
    }
  }
  return static_cast<double>(n) / k;
}
}  // namespace spatt