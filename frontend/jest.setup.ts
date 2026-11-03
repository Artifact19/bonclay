import "@testing-library/jest-dom";

// jsdom has no IntersectionObserver; Motion's whileInView / useInView need one. A no-op never reports "in view", which is
// fine: the server-rendered markup is complete. Tests that need to drive it install their own.
if (typeof window !== "undefined" && !("IntersectionObserver" in window)) {
  class NoopIntersectionObserver {
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds = [];
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  Object.defineProperty(window, "IntersectionObserver", { writable: true, configurable: true, value: NoopIntersectionObserver });
}
