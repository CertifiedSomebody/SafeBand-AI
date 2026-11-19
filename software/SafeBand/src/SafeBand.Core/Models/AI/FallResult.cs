using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.AI;

public sealed record FallResult : AiResult
{
    public bool IsFallDetected { get; init; }

    public double FallProbability { get; init; }
}
