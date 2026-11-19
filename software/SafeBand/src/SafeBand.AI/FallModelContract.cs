using System.Security.Cryptography;
using System.Text.RegularExpressions;

namespace SafeBand.AI;

public enum ModelReleaseStage
{
    DevelopmentOnly = 0,
    IndependentlyValidated = 1
}

public sealed record FallModelContract
{
    public required string ModelId { get; init; }

    public required string Version { get; init; }

    public required string FeatureSchemaVersion { get; init; }

    public required string ArtifactSha256 { get; init; }

    public required int IndependentSourceSessionCount { get; init; }

    public required ModelReleaseStage ReleaseStage { get; init; }

    public required bool IndependentEvaluationPassed { get; init; }

    public required bool HardwareValidationPassed { get; init; }

    public bool IsProductionEligible =>
        !string.IsNullOrWhiteSpace(ModelId)
        && !string.IsNullOrWhiteSpace(Version)
        && !string.IsNullOrWhiteSpace(FeatureSchemaVersion)
        && Regex.IsMatch(ArtifactSha256, "^[0-9a-fA-F]{64}$")
        && IndependentSourceSessionCount >= 3
        && ReleaseStage == ModelReleaseStage.IndependentlyValidated
        && IndependentEvaluationPassed
        && HardwareValidationPassed;

    public bool CanServeProduction(ReadOnlySpan<byte> artifact) =>
        IsProductionEligible && ArtifactMatches(artifact);

    public bool ArtifactMatches(ReadOnlySpan<byte> artifact) =>
        Regex.IsMatch(ArtifactSha256, "^[0-9a-fA-F]{64}$")
        && CryptographicOperations.FixedTimeEquals(
            Convert.FromHexString(ArtifactSha256),
            SHA256.HashData(artifact));
}
