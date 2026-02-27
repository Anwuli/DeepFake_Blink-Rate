import os
import subprocess
import argparse
import yaml
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

class BlinkPatternAnalyzer:
    def __init__(self, config_path="config.yaml"):
        """Initialize analyzer with configuration"""
        self.config = self.load_config(config_path)
        self.au_columns = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r',
                         ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r',
                         ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r',
                         ' AU26_r', ' AU45_r']
        
    def load_config(self, config_path):
        """Load and validate configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Set defaults
        config.setdefault('processing', {})
        config['processing'].setdefault('max_videos', 30)
        config['processing'].setdefault('skip_failures', True)
        
        # Set blink defaults if not present
        if 'blink' not in config:
            config['blink'] = {}
        config['blink'].setdefault('au45_threshold', 3.0)
        config['blink'].setdefault('min_blink_frames', 2)
        config['blink'].setdefault('max_blink_frames', 8)
        config['blink'].setdefault('human_blink_range', [8, 21])
        
        return config
    
    def process_video(self, video_path, output_dir=None):
        """Run OpenFace feature extraction on a single video"""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(video_path), "openface_output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            self.config['openface']['path'],
            "-f", os.path.abspath(video_path),
            "-out_dir", os.path.abspath(output_dir)
        ] + self.config['openface']['params']
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            csv_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".csv")
            return csv_path
        except subprocess.CalledProcessError as e:
            error_msg = f"OpenFace failed for {os.path.basename(video_path)}:\n"
            error_msg += f"Command: {' '.join(e.cmd)}\n"
            error_msg += f"Error code: {e.returncode}\n"
            error_msg += f"Output: {e.stdout}\n" if e.stdout else ""
            error_msg += f"Stderr: {e.stderr}\n" if e.stderr else ""
            raise RuntimeError(error_msg)

    def detect_blinks(self, df, fps=30):
        """Detect blink events from facial action units"""
        au45 = savgol_filter(df[' AU45_r'].values, 5, 2)
        confidence = df[' AU45_c'].values
        
        # Threshold detection
        blink_mask = (au45 > self.config['blink']['au45_threshold']) & (confidence > 0.8)
        changes = np.diff(blink_mask.astype(int))
        
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        # Handle edge cases
        if blink_mask[0]:
            starts = np.insert(starts, 0, 0)
        if blink_mask[-1]:
            ends = np.append(ends, len(blink_mask) - 1)
            
        # Filter by duration
        blinks = []
        min_frames = self.config['blink']['min_blink_frames']
        max_frames = self.config['blink']['max_blink_frames']
        
        for s, e in zip(starts, ends):
            duration = e - s
            if min_frames <= duration <= max_frames:
                blinks.append({
                    'start_frame': s,
                    'end_frame': e,
                    'duration': duration / fps,
                    'intensity': au45[s:e].mean()
                })
        
        return blinks

    def calculate_blink_stats(self, blinks, total_frames, fps):
        """Calculate statistics from detected blinks"""
        if not blinks:
            return {
                'blink_rate': 0,
                'avg_duration': 0,
                'duration_std': 0,
                'avg_intensity': 0,
                'interblink_intervals': [],
                'blink_count': 0
            }
        
        durations = [b['duration'] for b in blinks]
        intensities = [b['intensity'] for b in blinks]
        
        # Calculate inter-blink intervals
        intervals = []
        for i in range(1, len(blinks)):
            intervals.append((blinks[i]['start_frame'] - blinks[i-1]['end_frame']) / fps)
            
        # Calculate blink rate (blinks per minute)
        video_duration = total_frames / fps
        blink_rate = len(blinks) / video_duration * 60
        
        return {
            'blink_rate': blink_rate,
            'avg_duration': np.mean(durations),
            'duration_std': np.std(durations),
            'avg_intensity': np.mean(intensities),
            'interblink_intervals': intervals,
            'blink_count': len(blinks),
            'blink_frequency': 1 / (np.mean(intervals) if intervals else float('inf'))
        }

    def plot_analysis(self, df, blinks, blink_stats, output_path):
        """Generate analysis visualization"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Blink timeline - removed green highlighting
        plt.subplot(2, 2, 1)
        plt.plot(df['frame'], df[' AU45_r'], label='Eye Closure (AU45)')
        plt.axhline(y=self.config['blink']['au45_threshold'], 
                   color='r', linestyle='--', label='Threshold')
        
        # No more green highlighting for detected blinks
        # Just marking start and end points with vertical lines
        for blink in blinks:
            plt.axvline(x=blink['start_frame'], color='blue', alpha=0.5, linewidth=1)
            plt.axvline(x=blink['end_frame'], color='blue', alpha=0.5, linewidth=1)
        
        plt.xlabel('Frame Number')
        plt.ylabel('AU45 Intensity')
        plt.title('Blink Detection Timeline')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Blink rate comparison with typical range
        plt.subplot(2, 2, 2)
        human_range = self.config['blink']['human_blink_range']
        plt.bar(['Detected'], [blink_stats['blink_rate']], color='blue')
        plt.axhspan(human_range[0], human_range[1], color='green', alpha=0.3, 
                   label=f'Typical range ({human_range[0]}-{human_range[1]} blinks/min)')
        plt.title('Blink Rate')
        plt.ylabel('Blinks per minute')
        plt.legend()
        
        # Plot 3: Blink duration distribution
        plt.subplot(2, 2, 3)
        if blinks:
            durations = [b['duration'] for b in blinks]
            plt.hist(durations, bins=10, color='orange')
            plt.axvline(x=0.1, color='r', linestyle='--', 
                       label='Typical blink (100ms)')
            plt.axvline(x=0.4, color='g', linestyle='--', 
                       label='Extended blink (400ms)')
            plt.title('Blink Duration Distribution')
            plt.xlabel('Duration (seconds)')
            plt.legend()
        
        # Plot 4: Inter-blink intervals
        plt.subplot(2, 2, 4)
        if blink_stats['interblink_intervals']:
            plt.hist(blink_stats['interblink_intervals'], bins=10, color='purple')
            plt.title('Inter-blink Interval Distribution')
            plt.xlabel('Time between blinks (seconds)')
            
            # Add average line
            avg_interval = np.mean(blink_stats['interblink_intervals'])
            plt.axvline(x=avg_interval, color='r', linestyle='--', 
                       label=f'Avg: {avg_interval:.2f}s')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Not enough blinks to calculate intervals", 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Inter-blink Interval Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def analyze_video(self, video_path, output_dir="results"):
        """Complete analysis pipeline for a single video"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Process video with OpenFace
            csv_path = self.process_video(video_path)
            df = pd.read_csv(csv_path)
            
            # Analyze blinks
            fps = 30  # Adjust if your video has different FPS
            blinks = self.detect_blinks(df, fps)
            blink_stats = self.calculate_blink_stats(blinks, len(df), fps)
            
            # Generate output files
            self.plot_analysis(df, blinks, blink_stats, 
                             os.path.join(output_dir, 'analysis.png'))
            
            # Save text report
            with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
                f.write(f"Blink Pattern Analysis Report\n{'='*30}\n\n")
                f.write(f"Video: {os.path.basename(video_path)}\n")
                f.write(f"Total blinks detected: {blink_stats['blink_count']}\n")
                f.write(f"Blink rate: {blink_stats['blink_rate']:.2f} blinks/min\n")
                f.write(f"Average blink duration: {blink_stats['avg_duration']*1000:.1f} ms\n")
                f.write(f"Standard deviation of duration: {blink_stats['duration_std']*1000:.1f} ms\n")
                f.write(f"Average blink intensity: {blink_stats['avg_intensity']:.2f}\n")
                
                if blink_stats['interblink_intervals']:
                    avg_interval = np.mean(blink_stats['interblink_intervals'])
                    f.write(f"Average time between blinks: {avg_interval:.2f} seconds\n")
                    f.write(f"Blink frequency: {blink_stats['blink_frequency']:.2f} Hz\n")
                
                f.write(f"\nTypical blink rate range: {self.config['blink']['human_blink_range'][0]}-{self.config['blink']['human_blink_range'][1]} blinks/min\n")
                f.write(f"Typical blink duration: 100-400 ms\n")
            
            return {
                'video_path': video_path,
                'blinks': blinks,
                'blink_stats': blink_stats,
                'output_dir': output_dir,
                'success': True
            }
            
        except Exception as e:
            return {
                'video_path': video_path,
                'error': str(e),
                'success': False
            }

    def process_videos(self, folder_path="celeb-synthesis", output_base="results"):
        """Process multiple videos with progress tracking"""
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(glob(os.path.join(folder_path, ext)))
        
        if not videos:
            raise FileNotFoundError(f"No videos found in {folder_path}")
        
        # Sort and limit to max_videos
        videos = sorted(videos)[:self.config['processing']['max_videos']]
        results = []
        
        print(f"\nProcessing {len(videos)} videos from {folder_path}")
        
        for video_path in tqdm(videos, desc="Analyzing videos"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(output_base, video_name)
            
            result = self.analyze_video(video_path, output_dir)
            results.append(result)
            
            if not result['success'] and not self.config['processing']['skip_failures']:
                raise RuntimeError(f"Failed to process {video_name}: {result['error']}")
        
        return results

    def generate_summary_report(self, results, output_dir="claude_results"):
        """Generate consolidated summary of all analyses"""
        summary_path = os.path.join(output_dir, "claude_summary_report.txt")
        successful = [r for r in results if r['success']]
        
        with open(summary_path, 'w') as f:
            f.write(f"Blink Analysis Summary Report\n{'='*30}\n\n")
            f.write(f"Total videos processed: {len(results)}\n")
            f.write(f"Successfully analyzed: {len(successful)}\n")
            f.write(f"Failed: {len(results) - len(successful)}\n\n")
            
            if successful:
                # Calculate average stats across videos
                avg_blink_rate = np.mean([r['blink_stats']['blink_rate'] for r in successful])
                avg_duration = np.mean([r['blink_stats']['avg_duration'] for r in successful])
                avg_count = np.mean([r['blink_stats']['blink_count'] for r in successful])
                
                f.write("Average Statistics Across All Videos:\n")
                f.write(f"- Average blink rate: {avg_blink_rate:.2f} blinks/min\n")
                f.write(f"- Average blink duration: {avg_duration*1000:.1f} ms\n")
                f.write(f"- Average blink count: {avg_count:.1f} blinks per video\n")
                
                # List statistics for each video
                f.write("\nIndividual Video Statistics:\n")
                for r in successful:
                    video_name = os.path.basename(r['video_path'])
                    stats = r['blink_stats']
                    f.write(f"\n- {video_name}:\n")
                    f.write(f"  Blinks: {stats['blink_count']}\n")
                    f.write(f"  Rate: {stats['blink_rate']:.2f} blinks/min\n")
                    f.write(f"  Avg Duration: {stats['avg_duration']*1000:.1f} ms\n")
                
                # List failed videos
                if len(results) > len(successful):
                    f.write("\nFailed Videos:\n")
                    for r in results:
                        if not r['success']:
                            f.write(f"- {os.path.basename(r['video_path'])}: {r['error']}\n")
            
            print(f"\nSummary report saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Blink Pattern Analysis Tool")
    parser.add_argument("--folder", "-f", default="celeb-synthesis", 
                      help="Folder containing videos to analyze")
    parser.add_argument("--output", "-o", default="claude_results", 
                      help="Output directory for results")
    parser.add_argument("--config", "-c", default="config.yaml", 
                      help="Path to config file")
    args = parser.parse_args()
    
    try:
        analyzer = BlinkPatternAnalyzer(args.config)
        results = analyzer.process_videos(args.folder, args.output)
        analyzer.generate_summary_report(results, args.output)
        
        # Print quick summary to console
        successful = [r for r in results if r['success']]
        print(f"\nProcessing complete. {len(successful)}/{len(results)} videos analyzed successfully")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()