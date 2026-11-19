namespace SafeBand.Core.Models.Fusion;

public sealed record RiskContributor
{
    public required string Source { get; init; }

    public required string Signal { get; init; }

    public double Contribution { get; init; }

    public string? Explanation { get; init; }
}
