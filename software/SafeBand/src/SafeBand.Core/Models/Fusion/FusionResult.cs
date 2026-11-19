using SafeBand.Core.Enums;

namespace SafeBand.Core.Models.Fusion;

public sealed record FusionResult
{
    public RiskLevel RiskLevel { get; init; } = RiskLevel.Unknown;

    public double RiskScore { get; init; }

    public IReadOnlyList<RiskContributor> Contributors { get; init; }
        = Array.Empty<RiskContributor>();

    public DateTimeOffset? EvaluationTimeUtc { get; init; }
}
