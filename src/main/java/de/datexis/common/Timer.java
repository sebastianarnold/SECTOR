package de.datexis.common;

import com.google.common.base.Stopwatch;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper class to measure time
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Timer {

  protected final static Logger log = LoggerFactory.getLogger(Timer.class);
  
  private final Stopwatch timer;
  long currentSplit = 0;
  final Map<String,Long> splits;
  
  public static NumberFormat format = NumberFormat.getIntegerInstance(Locale.GERMAN);
  
  public Timer() {
    timer = Stopwatch.createUnstarted();
    splits = new TreeMap<>();
  }
  
  /**
   * Resets and starts the Timer. Splits are cleared.
   */
  public void start() {
    timer.reset();
    splits.clear();
    currentSplit = 0;
    timer.start();
  }
  
  /**
   * Stops the timer.
   * @return elapsed total time
   */
  public long stop() {
    try {
      timer.stop();
      return getLong();
    } catch(IllegalStateException ex) {
      log.warn(ex.toString());
      return 0;
    }
  }
  
  /**
   * @return elapsed total time
   */
  public long getLong() {
    return timer.elapsed(TimeUnit.MILLISECONDS);
  }
  
  /**
   * @param split
   * @return elapsed time for given split
   */
  public long getLong(String split) {
    if(!splits.containsKey(split)) {
      log.warn("Split " + split + " not found.");
      return 0;
    }
    return splits.get(split);
  }
  
  /**
   * @return string representation of total elapsed time (including "ms")
   */
  public String get() {
    return format.format(getLong()) + "ms";
  }
  
  /**
   * @param split
   * @return string representation of elapsed split time (including "ms")
   */
  public String get(String split) {
     return format.format(getLong(split)) + "ms";
  }
  
  /**
   * Starts a new split.
   * @param split name of the split
   * @return elapsed time from last split
   */
  public long setSplit(String split) {
    long s = resetSplit();
    splits.put(split, s);
    return s;
  }
  
  /**
   * Resets the current split.
   * @return elapsed time from current split befor reset
   */
  public long resetSplit() {
    long s = currentSplit;
    currentSplit = getLong();
    return currentSplit - s;
  }
  
  /**
   * converts time (in milliseconds) to human-readable format
   *  "<w> days, <x> hours, <y> minutes and (z) seconds"
   */
  public static String millisToLongDHMS(long duration) {
    
    final long ONE_SECOND = 1000;
    final long SECONDS = 60;
    final long ONE_MINUTE = ONE_SECOND * 60;
    final long MINUTES = 60;
    final long ONE_HOUR = ONE_MINUTE * 60;
    final long HOURS = 24;
    final long ONE_DAY = ONE_HOUR * 24;
    
    StringBuffer res = new StringBuffer();
    long temp = 0;
    if (duration >= ONE_MINUTE) {
      temp = duration / ONE_DAY;
      if (temp > 0) {
        duration -= temp * ONE_DAY;
        res.append(temp).append("d");
      }
      temp = duration / ONE_HOUR;
      if (temp > 0) {
        duration -= temp * ONE_HOUR;
        res.append(temp).append("h");
      }
      temp = duration / ONE_MINUTE;
      if (temp > 0) {
        duration -= temp * ONE_MINUTE;
        res.append(temp).append("m");
      }
      temp = duration / ONE_SECOND;
      if (temp > 0) {
        res.append(temp).append("s");
      }
      return res.toString();
    } else {
      return duration + "ms";
    }
  }
  
}
