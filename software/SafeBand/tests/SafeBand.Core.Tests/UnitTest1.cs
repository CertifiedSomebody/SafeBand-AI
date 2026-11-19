using SafeBand.Core.Enums;
using SafeBand.Core.Models;
using SafeBand.Core.Models.AI;
using SafeBand.Core.Models.Fusion;
using SafeBand.Core.Models.Quality;
using SafeBand.Core.Models.Sensors;

namespace SafeBand.Core.Tests;

public class CoreContractTests
{
    [Fact]
    public void ActivityState_ContainsExpectedValues()
    {
        Assert.Equal(0, (int)ActivityState.Unknown);
        Assert.Equal(1, (int)ActivityState.Stationary);
        Assert.Equal(2, (int)ActivityState.Walking);
        Assert.Equal(3, (int)ActivityState.Running);
        Assert.Equal(4, (int)ActivityState.Transition);
    }

    [Fact]
    public void DataSourceType_ContainsExpectedValues()
    {
        Assert.Equal(0, (int)DataSourceType.Unknown);
        Assert.Equal(1, (int)DataSourceType.Csv);
        Assert.Equal(2, (int)DataSourceType.Serial);
        Assert.Equal(3, (int)DataSourceType.Replay);
        Assert.Equal(4, (int)DataSourceType.LiveDevice);
    }

    [Fact]
    public void SensorValidity_ContainsExpectedValues()
    {
        Assert.Equal(0, (int)SensorValidity.Unknown);
        Assert.Equal(1, (int)SensorValidity.Valid);
        Assert.Equal(2, (int)SensorValidity.Invalid);
        Assert.Equal(3, (int)SensorValidity.Unavailable);
    }

    [Fact]
    public void TimingQuality_ContainsExpectedValues()
    {
        Assert.Equal(0, (int)TimingQuality.Unknown);
        Assert.Equal(1, (int)TimingQuality.Good);
        Assert.Equal(2, (int)TimingQuality.Irregular);
        Assert.Equal(3, (int)TimingQuality.Gap);
    }

    [Fact]
    public void AiResultStatus_ContainsExpectedValues()
    {
        Assert.Equal(0, (int)AiResultStatus.Unknown);
        Assert.Equal(1, (int)AiResultStatus.Valid);
        Assert.Equal(2, (int)AiResultStatus.InsufficientData);
        Assert.Equal(3, (int)AiResultStatus.InvalidInput);
        Assert.Equal(4, (int)AiResultStatus.Error);
    }

    [Fact]
    public void RiskLevel_ContainsExpectedValues()
    {
        Assert.Equal(0, (int)RiskLevel.Unknown);
        Assert.Equal(1, (int)RiskLevel.Normal);
        Assert.Equal(2, (int)RiskLevel.Elevated);
        Assert.Equal(3, (int)RiskLevel.High);
        Assert.Equal(4, (int)RiskLevel.Critical);
    }

    [Fact]
    public void SessionMetadata_StoresCoreMetadata()
    {
        var metadata = new SessionMetadata
        {
            SessionId = "session-001",
            SubjectId = "subject-001",
            Activity = ActivityState.Walking,
            Source = DataSourceType.Csv,
            StartTimeUtc = DateTimeOffset.UtcNow,
            ExpectedSampleIntervalMs = 200,
            OriginalFileName = "walking.csv"
        };

        Assert.Equal("session-001", metadata.SessionId);
        Assert.Equal("subject-001", metadata.SubjectId);
        Assert.Equal(ActivityState.Walking, metadata.Activity);
        Assert.Equal(DataSourceType.Csv, metadata.Source);
        Assert.Equal(200, metadata.ExpectedSampleIntervalMs);
        Assert.Equal("walking.csv", metadata.OriginalFileName);
    }

    [Fact]
    public void MotionSample_StoresMotionData()
    {
        var sample = new MotionSample
        {
            HeadingDeg = 10,
            RollDeg = 20,
            PitchDeg = 30,
            AccelXMetersPerSecondSquared = 1,
            AccelYMetersPerSecondSquared = 2,
            AccelZMetersPerSecondSquared = 9.8,
            GyroXRadiansPerSecond = 0.1,
            GyroYRadiansPerSecond = 0.2,
            GyroZRadiansPerSecond = 0.3
        };

        Assert.Equal(10, sample.HeadingDeg);
        Assert.Equal(20, sample.RollDeg);
        Assert.Equal(30, sample.PitchDeg);
        Assert.Equal(9.8, sample.AccelZMetersPerSecondSquared);
        Assert.Equal(0.3, sample.GyroZRadiansPerSecond);
    }

    [Fact]
    public void EnvironmentSample_StoresEnvironmentData()
    {
        var sample = new EnvironmentSample
        {
            TemperatureC = 34.5,
            HumidityPercent = 48,
            PressureHPa = 1011,
            GasKOhm = 110,
            Validity = SensorValidity.Valid
        };

        Assert.Equal(34.5, sample.TemperatureC);
        Assert.Equal(48, sample.HumidityPercent);
        Assert.Equal(1011, sample.PressureHPa);
        Assert.Equal(110, sample.GasKOhm);
        Assert.Equal(SensorValidity.Valid, sample.Validity);
    }

    [Fact]
    public void PpgSample_StoresPpgData()
    {
        var sample = new PpgSample
        {
            MaxIr = 100000,
            MaxRed = 80000,
            FingerDetected = true,
            RawHeartRateBpm = 120,
            RawHeartRateValid = true,
            RawSpo2Percent = 98,
            RawSpo2Valid = true,
            FinalHeartRateBpm = 121,
            FinalHeartRateValid = true,
            FinalSpo2Percent = 98.5,
            FinalSpo2Valid = true,
            Validity = SensorValidity.Valid
        };

        Assert.Equal(100000, sample.MaxIr);
        Assert.Equal(80000, sample.MaxRed);
        Assert.True(sample.FingerDetected);
        Assert.Equal(121, sample.FinalHeartRateBpm);
        Assert.Equal(98.5, sample.FinalSpo2Percent);
        Assert.Equal(SensorValidity.Valid, sample.Validity);
    }

    [Fact]
    public void AudioSample_StoresAudioData()
    {
        var sample = new AudioSample
        {
            Rms = 100,
            Dc = 2,
            AverageRms = 110,
            MaxRms = 250,
            Blocks = 32
        };

        Assert.Equal(100, sample.Rms);
        Assert.Equal(2, sample.Dc);
        Assert.Equal(110, sample.AverageRms);
        Assert.Equal(250, sample.MaxRms);
        Assert.Equal(32, sample.Blocks);
    }

    [Fact]
    public void PowerSample_StoresPowerData()
    {
        var sample = new PowerSample
        {
            BatteryVoltageV = 3.7,
            BatteryPercent = 50,
            ChargerConnected = true
        };

        Assert.Equal(3.7, sample.BatteryVoltageV);
        Assert.Equal(50, sample.BatteryPercent);
        Assert.True(sample.ChargerConnected);
    }

    [Fact]
    public void SampleQuality_StoresQualityInformation()
    {
        var quality = new SampleQuality
        {
            HasValidMotion = true,
            HasValidEnvironment = true,
            HasValidPpg = false,
            HasAudioData = true,
            HasPowerData = true,
            DeltaMilliseconds = 394,
            TimingWithinExpectedRange = false,
            ContainsUnavailableValues = true
        };

        Assert.True(quality.HasValidMotion);
        Assert.True(quality.HasValidEnvironment);
        Assert.False(quality.HasValidPpg);
        Assert.Equal(394, quality.DeltaMilliseconds);
        Assert.True(quality.ContainsUnavailableValues);
    }

    [Fact]
    public void TimingAssessment_StoresTimingInformation()
    {
        var assessment = new TimingAssessment
        {
            DeltaMilliseconds = 394,
            ExpectedIntervalMilliseconds = 200,
            Quality = TimingQuality.Irregular,
            EffectiveSampleRateHz = 1000.0 / 394.0
        };

        Assert.Equal(394, assessment.DeltaMilliseconds);
        Assert.Equal(200, assessment.ExpectedIntervalMilliseconds);
        Assert.Equal(TimingQuality.Irregular, assessment.Quality);
        Assert.NotNull(assessment.EffectiveSampleRateHz);
    }

    [Fact]
    public void SensorSample_CanContainAllSensorDomains()
    {
        var sample = new SensorSample
        {
            ElapsedMilliseconds = 394,
            SampleIndex = 1,
            Motion = new MotionSample
            {
                AccelZMetersPerSecondSquared = 9.8
            },
            Environment = new EnvironmentSample
            {
                TemperatureC = 34.5,
                Validity = SensorValidity.Valid
            },
            Ppg = new PpgSample
            {
                FingerDetected = true,
                FinalHeartRateBpm = 120,
                FinalHeartRateValid = true
            },
            Audio = new AudioSample
            {
                Rms = 100
            },
            Power = new PowerSample
            {
                BatteryVoltageV = 3.7
            },
            Quality = new SampleQuality
            {
                HasValidMotion = true,
                HasValidEnvironment = true,
                HasValidPpg = true,
                HasAudioData = true,
                HasPowerData = true
            }
        };

        Assert.Equal(394, sample.ElapsedMilliseconds);
        Assert.Equal(1, sample.SampleIndex);
        Assert.NotNull(sample.Motion);
        Assert.NotNull(sample.Environment);
        Assert.NotNull(sample.Ppg);
        Assert.NotNull(sample.Audio);
        Assert.NotNull(sample.Power);
        Assert.NotNull(sample.Quality);
    }

    [Fact]
    public void SensorSession_StoresMetadataAndSamples()
    {
        var session = new SensorSession
        {
            Metadata = new SessionMetadata
            {
                SessionId = "test-session",
                Activity = ActivityState.Stationary,
                Source = DataSourceType.Csv
            }
        };

        session.Samples.Add(new SensorSample
        {
            ElapsedMilliseconds = 0,
            SampleIndex = 0
        });

        Assert.Equal("test-session", session.Metadata.SessionId);
        Assert.Single(session.Samples);
        Assert.Equal(0, session.Samples[0].SampleIndex);
    }

    [Fact]
    public void ActivityResult_StoresPrediction()
    {
        var result = new ActivityResult
        {
            Status = AiResultStatus.Valid,
            Confidence = 0.91,
            ModelId = "activity-model",
            ModelVersion = "1.0",
            Activity = ActivityState.Walking
        };

        Assert.Equal(AiResultStatus.Valid, result.Status);
        Assert.Equal(0.91, result.Confidence);
        Assert.Equal("activity-model", result.ModelId);
        Assert.Equal(ActivityState.Walking, result.Activity);
    }

    [Fact]
    public void FallResult_StoresPrediction()
    {
        var result = new FallResult
        {
            Status = AiResultStatus.Valid,
            Confidence = 0.97,
            IsFallDetected = true,
            FallProbability = 0.97
        };

        Assert.True(result.IsFallDetected);
        Assert.Equal(0.97, result.FallProbability);
        Assert.Equal(AiResultStatus.Valid, result.Status);
    }

    [Fact]
    public void HeartRateResult_StoresPhysiologicalPrediction()
    {
        var result = new HeartRateResult
        {
            Status = AiResultStatus.Valid,
            Confidence = 0.88,
            HeartRateBpm = 124,
            Spo2Percent = 98,
            FingerDetected = true
        };

        Assert.Equal(124, result.HeartRateBpm);
        Assert.Equal(98, result.Spo2Percent);
        Assert.True(result.FingerDetected);
    }

    [Fact]
    public void AudioResult_StoresAudioPrediction()
    {
        var result = new AudioResult
        {
            Status = AiResultStatus.Valid,
            Confidence = 0.84,
            EventLabel = "Impact",
            EventProbability = 0.84
        };

        Assert.Equal("Impact", result.EventLabel);
        Assert.Equal(0.84, result.EventProbability);
    }

    [Fact]
    public void RiskContributor_StoresContribution()
    {
        var contributor = new RiskContributor
        {
            Source = "FallModel",
            Signal = "FallProbability",
            Contribution = 0.72,
            Explanation = "High fall probability detected."
        };

        Assert.Equal("FallModel", contributor.Source);
        Assert.Equal("FallProbability", contributor.Signal);
        Assert.Equal(0.72, contributor.Contribution);
        Assert.Equal("High fall probability detected.", contributor.Explanation);
    }

    [Fact]
    public void FusionResult_StoresRiskAndContributors()
    {
        var result = new FusionResult
        {
            RiskLevel = RiskLevel.High,
            RiskScore = 0.82,
            Contributors =
            [
                new RiskContributor
                {
                    Source = "FallModel",
                    Signal = "FallProbability",
                    Contribution = 0.72
                },
                new RiskContributor
                {
                    Source = "ActivityModel",
                    Signal = "Activity",
                    Contribution = 0.10
                }
            ],
            EvaluationTimeUtc = DateTimeOffset.UtcNow
        };

        Assert.Equal(RiskLevel.High, result.RiskLevel);
        Assert.Equal(0.82, result.RiskScore);
        Assert.Equal(2, result.Contributors.Count);
        Assert.Equal("FallModel", result.Contributors[0].Source);
    }
}
