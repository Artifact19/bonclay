import { AREA_HOME, areaForPath, readRememberedArea, rememberArea } from "@/lib/areas";

describe("areaForPath: an explicit map, longest entry wins", () => {
  test.each([
    ["/dashboard", "agents"],
    ["/dashboard/studio/agents", "agents"],
    ["/dashboard/studio/agents/", "agents"],
    ["/dashboard/calls", "agents"],
    ["/dashboard/campaigns", "agents"],
    ["/dashboard/approvals", "agents"],
    ["/dashboard/playground", "agents"],
    ["/dashboard/knowledge", "agents"],
    ["/dashboard/knowledge/try", "agents"],
    ["/dashboard/analytics", "agents"],
    ["/dashboard/eval", "agents"],
    ["/dashboard/studio", "studio"],
    ["/dashboard/studio/home", "studio"],
    ["/dashboard/say-it-once", "studio"],
  ])("%s belongs to %s", (path, area) => expect(areaForPath(path)).toBe(area));

  test.each(["/dashboard/voices", "/dashboard/keys", "/verify", "/dashboard/something-new", "/", "", null])(
    "%s is shared, so the current area stays",
    (path) => expect(areaForPath(path)).toBeNull()
  );

  test("matches whole path pieces, not text prefixes", () => {
    expect(areaForPath("/dashboard/studiofoo")).toBeNull();
    expect(areaForPath("/dashboard/callsign")).toBeNull();
  });

  test("each area has its own home", () => {
    expect(AREA_HOME).toEqual({ studio: "/dashboard/studio/home", agents: "/dashboard" });
    expect(areaForPath(AREA_HOME.studio)).toBe("studio");
    expect(areaForPath(AREA_HOME.agents)).toBe("agents");
  });
});

describe("remembered area", () => {
  beforeEach(() => localStorage.clear());

  test("round trip, ignoring junk", () => {
    expect(readRememberedArea()).toBeNull();
    rememberArea("studio");
    expect(readRememberedArea()).toBe("studio");
    localStorage.setItem("bonclay.area", "nonsense");
    expect(readRememberedArea()).toBeNull();
  });

  test("blocked storage never throws", () => {
    const get = jest.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    const set = jest.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    expect(readRememberedArea()).toBeNull();
    expect(() => rememberArea("agents")).not.toThrow();
    get.mockRestore();
    set.mockRestore();
  });
});
