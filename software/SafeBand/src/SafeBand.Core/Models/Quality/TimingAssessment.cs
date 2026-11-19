using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.Quality;

public sealed record TimingAssessment
{
    public long? DeltaMilliseconds { get; init; }

    public int ExpectedIntervalMilliseconds { get; init; }

    public TimingQuality Quality { get; init; } = TimingQuality.Unknown;

    public double? EffectiveSampleRateHz { get; init; }
}
