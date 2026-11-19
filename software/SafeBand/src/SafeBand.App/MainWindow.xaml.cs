using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

using System.Diagnostics;
using System.IO;
using IOPath = System.IO.Path;

namespace SafeBand.App;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private void LaunchButton_Click(object sender, RoutedEventArgs e)
    {
        var root = FindRepositoryRoot(AppContext.BaseDirectory);
        if (root is null)
        {
            StatusText.Text = "Unable to locate repository root.";
            return;
        }

        var pythonCandidates = new[]
        {
            IOPath.Combine(root, "venv", "Scripts", "python.exe"),
            "python"
        };
        var python = pythonCandidates.FirstOrDefault(candidate => File.Exists(candidate)) ?? "python";
        var app = IOPath.Combine(root, "dashboard", "replay_app.py");

        var startInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = $"-m streamlit run \"{app}\" --server.headless=true --server.port=8501",
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        try
        {
            Process.Start(startInfo);
            Process.Start(new ProcessStartInfo
            {
                FileName = "http://localhost:8501",
                UseShellExecute = true
            });
            StatusText.Text = "Launching replay dashboard...";
        }
        catch (System.Exception ex)
        {
            StatusText.Text = $"Launch failed: {ex.Message}";
        }
    }

    private static string? FindRepositoryRoot(string start)
    {
        var directory = new DirectoryInfo(start);
        while (directory is not null)
        {
            if (File.Exists(IOPath.Combine(directory.FullName, "dashboard", "replay_app.py")))
                return directory.FullName;
            directory = directory.Parent;
        }
        return null;
    }
}