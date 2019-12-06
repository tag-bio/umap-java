package com.tagbio.umap;

import java.util.ArrayList;
import java.util.List;


public class UmapProgress {
  private static UmapProgress SINGLETON = null;
  private final static long MIN_UPDATE_PERIOD = 500; // milliseconds

  private List<ProgressListener> mProgressListeners = new ArrayList<>();
  private int mTotal = 0;
  private int mCounter = 0;
  private long mLastNotificationTime = 0L;

  private UmapProgress() {
  }

  private static UmapProgress get() {
    if (SINGLETON == null) {
      SINGLETON = new UmapProgress();
    }
    return SINGLETON;
  }

  public static void addProgressListener(final ProgressListener listener) {
    final UmapProgress progress = get();
    if (!progress.mProgressListeners.contains(listener)) {
      progress.mProgressListeners.add(listener);
    }
  }

  public static boolean removeProgressListener(final ProgressListener listener) {
    return get().mProgressListeners.remove(listener);
  }

  protected void notifyListeners(ProgressState state) {
    // limit calls to notify if occurring too often
    final long now = System.currentTimeMillis();
    if (now - mLastNotificationTime > MIN_UPDATE_PERIOD) {
      for (final ProgressListener listener : mProgressListeners) {
        listener.updated(state);
      }
      mLastNotificationTime = now;
    }
  }

  public static void reset(final int total) {
    final UmapProgress progress = get();
    progress.mTotal = total;
    progress.mCounter = 0;
    progress.mLastNotificationTime = 0L;
    progress.update(0);
  }

  public static void incTotal(final int inc) {
    final UmapProgress progress = get();
    progress.mTotal += inc;
    progress.update(0);
  }

  public static void finished() {
    final UmapProgress progress = get();
    progress.mCounter = progress.mTotal;
    progress.mLastNotificationTime = 0L;
    progress.update(0);
  }

  public static void update() {
    get().update(1);
  }

  public static void update(int n) {
    final UmapProgress progress = get();
    progress.mCounter += n;
    if (progress.mCounter > progress.mTotal) {
      Utils.message("Update counter exceeded total: " + progress.mCounter + " : " + progress.mTotal);
    }
    progress.notifyListeners(getProgress());
  }

  public static ProgressState getProgress() {
    final UmapProgress progress = get();
    return new ProgressState(progress.mTotal, progress.mCounter);
  }
}
