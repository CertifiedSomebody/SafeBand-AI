using SafeBand.Core.Models.Quality;
using SafeBand.Core.Models.Sensors;

namespace SafeBand.Core.Models;

public sealed record SensorSample
{
    public required long ElapsedMilliseconds { get; init; }

    public required long SampleIndex { get; init; }

    public MotionSample? Motion { get; init; }

    public EnvironmentSample? Environment { get; init; }

    public PpgSample? Ppg { get; init; }

    public AudioSample? Audio { get; init; }

    public PowerSample? Power { get; init; }

    public SampleQuality? Quality { get; init; }
}
