using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.Sensors;

public sealed record PpgSample
{
    public long MaxIr { get; init; }

    public long MaxRed { get; init; }

    public bool FingerDetected { get; init; }

    public double RawHeartRateBpm { get; init; }

    public bool RawHeartRateValid { get; init; }

    public double RawSpo2Percent { get; init; }

    public bool RawSpo2Valid { get; init; }

    public double FinalHeartRateBpm { get; init; }

    public bool FinalHeartRateValid { get; init; }

    public double FinalSpo2Percent { get; init; }

    public bool FinalSpo2Valid { get; init; }

    public SensorValidity Validity { get; init; } = SensorValidity.Unknown;
}
