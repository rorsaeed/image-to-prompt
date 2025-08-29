using CliWrap;
using CliWrap.Buffered;

using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Management;
using System.Net.Http;
using System.Reflection;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Ribbon;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace AiPromptAssistant
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private  BackgroundWorker _worker;
        private CommandResult? _commandResult;
        public MainWindow()
        {
            InitializeComponent();

            //var process = new Process();
            //process.StartInfo.FileName = "cmd.exe";
            //process.StartInfo.Arguments = "/c .\\python_embeded\\Scripts\\streamlit run app.py --server.headless true";
            //process.StartInfo.WorkingDirectory = AppDomain.CurrentDomain.BaseDirectory + "\\python";
            //process.StartInfo.UseShellExecute = false;
            //process.StartInfo.RedirectStandardOutput = true;
            //process.StartInfo.RedirectStandardError = true;

            //process.OutputDataReceived += (s, e) => Console.WriteLine(e.Data);
            //process.ErrorDataReceived += (s, e) => Console.Error.WriteLine(e.Data);

            //process.Start();
            //process.BeginOutputReadLine();
            //process.BeginErrorReadLine();
            //process.WaitForExit();
          
        }
        private void Worker_DoWork(object? sender, DoWorkEventArgs e)
        {
            try
            {
                var startupPath = AppDomain.CurrentDomain.BaseDirectory + "\\python";
                var call = Cli.Wrap("cmd.exe")
                    .WithArguments($"/c .\\python_embeded\\python.exe .\\python_embeded\\Scripts\\streamlit.exe run app.py --server.headless true")
                    .WithWorkingDirectory(startupPath)
                    .WithStandardOutputPipe(PipeTarget.ToDelegate(Console.WriteLine))
                    .WithStandardErrorPipe(PipeTarget.ToDelegate(Console.WriteLine))
                    .WithValidation(CommandResultValidation.None);

                // Execute synchronously since we're already in a background thread
                _commandResult = call.ExecuteBufferedAsync().GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                e.Result = ex;
            }
        }
        private async void Worker_RunWorkerCompleted(object? sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                MessageBox.Show($"Error executing command: {e.Error.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            if (e.Result is Exception ex)
            {
                MessageBox.Show($"Error executing command: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

         
        }
        private async void Window_Loaded(object sender, RoutedEventArgs e)
        {
            var startupPath = AppDomain.CurrentDomain.BaseDirectory;
            await ExtractAndDeleteAsync(Path.Combine(startupPath, "python.zip"), startupPath);
            _worker = new BackgroundWorker();
            _worker.DoWork += Worker_DoWork;
            _worker.RunWorkerCompleted += Worker_RunWorkerCompleted;
            _worker.WorkerSupportsCancellation = true;
            _worker.RunWorkerAsync();
          

            try
            {
                // Initialize and navigate WebView2 on UI thread
                await webView2.EnsureCoreWebView2Async();
                webView2.CoreWebView2.Navigate("http://localhost:8501");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error initializing WebView: {ex.Message}", "Error",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private static void KillProcessAndChildren(int pid)
        {
            // Cannot close 'system idle process'.
            if (pid == 0) return;
            var searcher = new ManagementObjectSearcher
                ("Select * From Win32_Process Where ParentProcessID=" + pid);
            var moc = searcher.Get();
            foreach (ManagementObject mo in moc) KillProcessAndChildren(Convert.ToInt32(mo["ProcessID"]));
            try
            {
                var proc = Process.GetProcessById(pid);
                proc.Kill();
            }
            catch (ArgumentException)
            {
                // Process already exited.
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {

            Hide();

            if (_worker.IsBusy)
            {
                _worker.CancelAsync();
            }

            try
            {
                var nProcessID = Process.GetCurrentProcess().Id;
                KillProcessAndChildren(nProcessID);
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception);
            }

        }

        private void MenuItem_Click(object sender, RoutedEventArgs e)
        {
            try
            {

                var url = "https://github.com/rorsaeed/image-to-prompt";
                BrowserHelper.OpenUrl(url);


                // Open the link in the default browser
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception);
            }
        }

        private void MenuItem_Click_1(object sender, RoutedEventArgs e)
        {
            try
            {

                var url = "https://eng.webphotogallery.store/anime/";
                BrowserHelper.OpenUrl(url);


                // Open the link in the default browser
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception);
            }
        }

        private void MenuItem_Click_2(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private async void MenuItem_Click_3(object sender, RoutedEventArgs e)
        {
           

        }

        public static async Task ExtractAndDeleteAsync(string zipPath, string extractTo)
        {
            if (!File.Exists(zipPath))
            {
                Console.WriteLine($"ZIP file not found: {zipPath}");
                return;
            }

            try
            {
                await Task.Run(() =>
                {
                    Directory.CreateDirectory(extractTo);
                    ZipFile.ExtractToDirectory(zipPath, extractTo, overwriteFiles: true);
                    File.Delete(zipPath);
                });

                Console.WriteLine("✅ Extraction complete and ZIP file deleted.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
        }


    }
}