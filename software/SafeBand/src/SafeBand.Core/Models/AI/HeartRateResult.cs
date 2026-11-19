using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.AI;

public sealed record HeartRateResult : AiResult
{
    public double? HeartRateBpm { get; init; }

    public double? Spo2Percent { get; init; }

    public bool FingerDetected { get; init; }
}
