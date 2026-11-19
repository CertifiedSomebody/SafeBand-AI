using SafeBand.AI;
using SafeBand.Core.Models;
using SafeBand.Core.Models.Quality;
using SafeBand.Core.Models.Sensors;

namespace SafeBand.Core.Tests;

public sealed class TemporalFallStateMachineTests
{
    [Fact]
    public void ConfirmsOnlyAfterConfiguredConsecutiveHighProbabilities()
    {
        var machine = new TemporalFallStateMachine();

        Assert.Equal(
            FallState.Suspected,
            machine.Update(Sample(0), 0.9).State);
        Assert.False(machine.Update(Sample(10), 0.9).NewlyConfirmed);

        var confirmed = machine.Update(Sample(20), 0.9, "dev-0.1");

        Assert.Equal(FallState.Confirmed, confirmed.State);
        Assert.True(confirmed.NewlyConfirmed);
        Assert.Equal("dev-0.1", confirmed.ModelVersion);
    }

    [Fact]
    public void MissingMotionOrBadTimingCannotConfirmEvent()
    {
        var machine = new TemporalFallStateMachine();

        var decision = machine.Update(
            Sample(0, hasMotion: false),
            1.0);

        Assert.Equal(FallState.InsufficientData, decision.State);
        Assert.False(decision.NewlyConfirmed);
        Assert.False(decision.HasValidMotionAndTiming);
    }

    [Fact]
    public void MissingPpgDoesNotBlockMotionBasedStateMachine()
    {
        var machine = new TemporalFallStateMachine();
        TemporalFallDecision? result = null;

        for (var index = 0; index < 3; index++)
        {
            result = machine.Update(Sample(index * 10), 0.9);
        }

        Assert.NotNull(result);
        Assert.Equal(FallState.Confirmed, result.State);
    }

    [Fact]
    public void ConfirmationIsLatchedUntilRecoveryAndCooldown()
    {
        var machine = new TemporalFallStateMachine(
            recoverySamples: 2,
            recoveryCooldownMilliseconds: 100);
        for (var index = 0; index < 3; index++)
        {
            machine.Update(Sample(index * 10), 0.9);
        }

        Assert.False(machine.Update(Sample(30), 0.1).NewlyConfirmed);
        Assert.Equal(FallState.Confirmed, machine.Update(Sample(40), 0.1).State);
        Assert.Equal(FallState.Recovery, machine.Update(Sample(50), 0.1).State);
        Assert.False(machine.Update(Sample(100), 0.9).NewlyConfirmed);
        Assert.Equal(FallState.Confirmed, machine.Update(Sample(110), 0.9).State);
        Assert.Equal(FallState.Confirmed, machine.Update(Sample(120), 0.1).State);
        Assert.Equal(FallState.Confirmed, machine.Update(Sample(130), 0.1).State);
        Assert.Equal(FallState.Recovery, machine.Update(Sample(140), 0.1).State);
        Assert.Equal(FallState.Monitoring, machine.Update(Sample(230), 0.1).State);
    }

    [Fact]
    public void RejectsNonMonotonicElapsedTime()
    {
        var machine = new TemporalFallStateMachine();
        machine.Update(Sample(10), 0.1);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => machine.Update(Sample(10), 0.1));
    }

    [Fact]
    public void ModelContractBlocksDevelopmentModelsAndInsufficientLineage()
    {
        var contract = new FallModelContract
        {
            ModelId = "fall-detector",
            Version = "0.1.0",
            FeatureSchemaVersion = "1.0",
            ArtifactSha256 = new string('a', 64),
            IndependentSourceSessionCount = 1,
            ReleaseStage = ModelReleaseStage.DevelopmentOnly,
            IndependentEvaluationPassed = false,
            HardwareValidationPassed = false
        };

        Assert.False(contract.IsProductionEligible);
        Assert.False(contract.CanServeProduction("not the artifact"u8));
        Assert.False(contract.ArtifactMatches("not the artifact"u8));
    }

    private static SensorSample Sample(
        long elapsedMilliseconds,
        bool hasMotion = true,
        bool timingWithinExpectedRange = true) =>
        new()
        {
            ElapsedMilliseconds = elapsedMilliseconds,
            SampleIndex = elapsedMilliseconds / 10,
            Motion = hasMotion
                ? new MotionSample()
                : null,
            Quality = new SampleQuality
            {
                HasValidMotion = hasMotion,
                TimingWithinExpectedRange = timingWithinExpectedRange,
                DeltaMilliseconds = elapsedMilliseconds == 0 ? null : 10
            }
        };
}
