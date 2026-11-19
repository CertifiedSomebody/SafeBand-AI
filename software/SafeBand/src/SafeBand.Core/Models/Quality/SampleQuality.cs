namespace SafeBand.Core.Models.Quality;

public sealed record SampleQuality
{
    public bool HasValidMotion { get; init; }

    public bool HasValidEnvironment { get; init; }

    public bool HasValidPpg { get; init; }

    public bool HasAudioData { get; init; }

    public bool HasPowerData { get; init; }

    public long? DeltaMilliseconds { get; init; }

    public bool TimingWithinExpectedRange { get; init; }

    public bool ContainsUnavailableValues { get; init; }
}
