// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mpi/record_exchange.h"
#include "framework/runtime.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace opensn;

namespace
{

/// Payload of the record with `key` sent by `source`: `key % 4` values derived from both.
std::vector<double>
MakePayload(const std::uint64_t key, const int source)
{
  std::vector<double> payload;
  for (std::uint64_t i = 0; i < key % 4; ++i)
    payload.push_back(static_cast<double>(1000 * source) + static_cast<double>(key) + 0.5 * i);
  return payload;
}

/// Where a FakeFileSet fails, on its failing rank.
enum class FailAt
{
  never,
  open,
  prepare,
  begin,
  fill,
  receive
};

/**
 * An in-memory file set for RunFileSetPasses: file k holds records with keys 10 k + i for
 * i < k % 3 (so some files are empty), sent in two streams to rank key % P. Stream s carries the
 * payload {key + s}.
 */
class FakeFileSet
{
public:
  FakeFileSet(const std::uint64_t num_files, const FailAt fail_at, const int failing_rank)
    : num_files_(num_files), fail_at_(fail_at), failing_rank_(failing_rank)
  {
  }

  static std::vector<std::uint64_t> FileKeys(const std::uint64_t file_id)
  {
    std::vector<std::uint64_t> keys;
    for (std::uint64_t i = 0; i < file_id % 3; ++i)
      keys.push_back(10 * file_id + i);
    return keys;
  }

  void OpenFile(const std::uint64_t file_id)
  {
    Fail(FailAt::open);
    keys_ = FileKeys(file_id);
  }

  std::string PrepareFile(const bool has_file)
  {
    // A collective step on every rank, as a real source's owner lookup is.
    int files = has_file ? 1 : 0;
    mpi_comm.all_reduce(files, mpi::op::sum<int>());
    if (fail_at_ == FailAt::prepare and mpi_comm.rank() == failing_rank_)
      return "injected prepare failure";
    return {};
  }

  std::size_t GetHeaderWidth(std::size_t /*stream*/) const { return 1; }

  void BeginStream(std::size_t /*stream*/)
  {
    Fail(FailAt::begin);
    next_ = 0;
  }

  bool IsStreamDone(std::size_t /*stream*/) const { return next_ == keys_.size(); }

  void FillChunk(const std::size_t stream, std::uint64_t /*budget*/, RecordBatch& batch)
  {
    Fail(FailAt::fill);
    // One record per chunk, the smallest the budget rule allows.
    const auto key = keys_[next_++];
    batch.AddRecord(static_cast<int>(key % static_cast<std::uint64_t>(mpi_comm.size())))[0] =
      static_cast<std::int64_t>(key);
    const double value = static_cast<double>(key + stream);
    batch.AppendPayload(&value, 1);
  }

  void Receive(const std::size_t stream, const std::uint64_t pass, const ReceivedRecords& records)
  {
    if (records.GetNumRecords() > 0)
      Fail(FailAt::receive);
    for (std::size_t r = 0; r < records.GetNumRecords(); ++r)
    {
      const auto key = static_cast<std::uint64_t>(records.GetHeader(r)[0]);
      const auto file_id = pass * static_cast<std::uint64_t>(mpi_comm.size()) +
                           static_cast<std::uint64_t>(records.sources[r]);
      received_.emplace_back(stream, key, file_id, records.GetPayload(r)[0]);
    }
  }

  void CloseFile() { keys_.clear(); }

  /// (stream, key, file, payload) of every record received.
  const std::vector<std::tuple<std::size_t, std::uint64_t, std::uint64_t, double>>&
  GetReceived() const
  {
    return received_;
  }

private:
  void Fail(const FailAt where) const
  {
    if (fail_at_ == where and mpi_comm.rank() == failing_rank_)
      throw std::runtime_error("injected failure");
  }

  const std::uint64_t num_files_;
  const FailAt fail_at_;
  const int failing_rank_;
  std::vector<std::uint64_t> keys_;
  std::size_t next_ = 0;
  std::vector<std::tuple<std::size_t, std::uint64_t, std::uint64_t, double>> received_;
};

} // namespace

TEST(RecordExchangeTest, DeliversRecordsToTheirOwners)
{
  const auto rank = mpi_comm.rank();
  const auto num_ranks = mpi_comm.size();

  // Rank r sends keys r, r + P, ..., some to itself, with payloads of length 0 to 3. Rank 0 sends
  // nothing, so a rank may have no records.
  RecordBatch batch(2);
  std::vector<std::uint64_t> sent_keys;
  if (rank > 0)
    for (std::uint64_t key = static_cast<std::uint64_t>(rank); key < 40;
         key += static_cast<std::uint64_t>(num_ranks))
      sent_keys.push_back(key);
  for (const auto key : sent_keys)
  {
    auto* header = batch.AddRecord(static_cast<int>((key * 7) % num_ranks));
    header[0] = static_cast<std::int64_t>(key);
    header[1] = -static_cast<std::int64_t>(key);
    const auto payload = MakePayload(key, rank);
    batch.AppendPayload(payload.data(), payload.size());
  }
  const auto received = ExchangeRecords(mpi_comm, std::move(batch), "Test exchange");
  EXPECT_EQ(batch.GetNumRecords(), 0U);

  // Every key with (7 key) mod P == rank arrives once, from the rank that sent it.
  std::set<std::uint64_t> expected;
  for (int source = 1; source < num_ranks; ++source)
    for (std::uint64_t key = static_cast<std::uint64_t>(source); key < 40;
         key += static_cast<std::uint64_t>(num_ranks))
      if (static_cast<int>((key * 7) % num_ranks) == rank)
        expected.insert(key);
  ASSERT_EQ(received.GetNumRecords(), expected.size());
  std::set<std::uint64_t> seen;
  int previous_source = 0;
  for (std::size_t r = 0; r < received.GetNumRecords(); ++r)
  {
    const auto* header = received.GetHeader(r);
    const auto key = static_cast<std::uint64_t>(header[0]);
    EXPECT_EQ(header[1], -header[0]);
    EXPECT_EQ(received.sources[r], static_cast<int>(key % num_ranks));
    // Arrival order is by source rank.
    EXPECT_GE(received.sources[r], previous_source);
    previous_source = received.sources[r];
    const auto payload = MakePayload(key, received.sources[r]);
    ASSERT_EQ(received.GetPayloadSize(r), payload.size());
    for (std::size_t i = 0; i < payload.size(); ++i)
      EXPECT_EQ(received.GetPayload(r)[i], payload[i]);
    EXPECT_TRUE(seen.insert(key).second);
  }
  EXPECT_EQ(seen, expected);
}

TEST(RecordExchangeTest, RejectsAnOwnerOutsideTheCommunicator)
{
  RecordBatch batch(1);
  if (mpi_comm.rank() == 0)
    batch.AddRecord(mpi_comm.size())[0] = 1;
  EXPECT_THROW(ExchangeRecords(mpi_comm, std::move(batch), "Test exchange"), std::logic_error);
}

TEST(RecordExchangeTest, OwnerDirectoryFindsRegisteredKeys)
{
  const auto rank = mpi_comm.rank();
  const auto num_ranks = static_cast<std::uint64_t>(mpi_comm.size());

  // Rank r owns keys 3 k + 1 with k mod P == r; every rank asks for keys 0 to 3 * 10 P.
  std::vector<std::uint64_t> owned;
  for (std::uint64_t k = static_cast<std::uint64_t>(rank); k < 10 * num_ranks; k += num_ranks)
    owned.push_back(3 * k + 1);
  const OwnerDirectory directory(mpi_comm, owned);

  std::vector<std::uint64_t> keys;
  for (std::uint64_t key = 0; key < 30 * num_ranks; ++key)
    keys.push_back(key);
  const auto owners = directory.FindOwners(keys);
  ASSERT_EQ(owners.size(), keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i)
  {
    const auto expected = keys[i] % 3 == 1 ? static_cast<int>(((keys[i] - 1) / 3) % num_ranks) : -1;
    EXPECT_EQ(owners[i], expected) << "key " << keys[i];
  }

  // A rank may ask for nothing while the others ask.
  EXPECT_TRUE(directory.FindOwners(rank == 0 ? std::vector<std::uint64_t>{} : keys).size() ==
              (rank == 0 ? 0 : keys.size()));
}

TEST(RecordExchangeTest, OwnerDirectoryRejectsDuplicateKeys)
{
  // Every rank registers key 5, and rank 0 registers it twice in a one-rank run.
  std::vector<std::uint64_t> owned = {5};
  if (mpi_comm.size() == 1)
    owned.push_back(5);
  EXPECT_THROW(OwnerDirectory(mpi_comm, owned), std::logic_error);
}

TEST(RecordExchangeTest, FileSetPassesDeliverEveryRecordOnce)
{
  const auto rank = static_cast<std::uint64_t>(mpi_comm.rank());
  const auto num_ranks = static_cast<std::uint64_t>(mpi_comm.size());
  for (const auto num_files : {std::uint64_t{1}, num_ranks + 1, 2 * num_ranks + 1, num_ranks})
  {
    FakeFileSet source(num_files, FailAt::never, 0);
    RunFileSetPasses<std::runtime_error>(mpi_comm, num_files, 2, 1, source, "Test file set");

    std::multiset<std::tuple<std::size_t, std::uint64_t, std::uint64_t, double>> expected, seen;
    for (std::uint64_t file_id = 0; file_id < num_files; ++file_id)
      for (const auto key : FakeFileSet::FileKeys(file_id))
        if (key % num_ranks == rank)
          for (std::size_t stream = 0; stream < 2; ++stream)
            expected.emplace(stream, key, file_id, static_cast<double>(key + stream));
    for (const auto& record : source.GetReceived())
      seen.insert(record);
    EXPECT_EQ(seen, expected) << num_files << " files";
  }
}

TEST(RecordExchangeTest, FileSetPassesRaiseAFailureOnEveryRank)
{
  // Rank P - 1 fails at each step in turn. Files 1 and 2 have records, and with 3 P files every
  // rank reads at least one file that has records.
  const auto num_files = 3 * static_cast<std::uint64_t>(mpi_comm.size());
  for (const auto fail_at :
       {FailAt::open, FailAt::prepare, FailAt::begin, FailAt::fill, FailAt::receive})
  {
    FakeFileSet source(num_files, fail_at, mpi_comm.size() - 1);
    EXPECT_THROW(
      RunFileSetPasses<std::runtime_error>(mpi_comm, num_files, 2, 1, source, "Test file set"),
      std::runtime_error);
    // Every rank left the passes, so a following collective completes.
    int count = 1;
    mpi_comm.all_reduce(count, mpi::op::sum<int>());
    EXPECT_EQ(count, mpi_comm.size());
  }
}
