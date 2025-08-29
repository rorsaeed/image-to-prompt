using System.Configuration;
using System.Data;
using System.Windows;

namespace AiPromptAssistant
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override async void OnStartup(StartupEventArgs e)
        {
            var splash = new SplashWindow();
            splash.Show();

            var timer = new System.Diagnostics.Stopwatch();
            timer.Start();

            base.OnStartup(e);
            var main = new MainWindow();

            timer.Stop();
            int minSplashTime = 1500;
            int remaining = minSplashTime - (int)timer.ElapsedMilliseconds;
            if (remaining > 0)await Task.Delay(remaining);

            splash.StartFadeOutAndClose();
            main.Show();
            
        }
    }

}
