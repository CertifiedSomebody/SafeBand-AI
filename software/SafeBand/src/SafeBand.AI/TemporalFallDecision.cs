namespace SafeBand.AI;

public sealed record TemporalFallDecision
{
    public required FallState State { get; init; }

    public double? FallProbability { get; init; }

    public bool NewlyConfirmed { get; init; }

    public bool HasValidMotionAndTiming { get; init; }

    public string? ModelVersion { get; init; }
}
