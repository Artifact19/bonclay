import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { createServerClient } from "@supabase/ssr";

let warned = false;

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (!pathname.startsWith("/dashboard")) {
    return NextResponse.next();
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  // Supabase not configured: this is a local app with no accounts, so the dashboard is open in any mode.
  if (!url || !key) {
    if (!warned) {
      console.warn("[bonclay] Supabase env vars missing; running in local mode (no sign-in required).");
      warned = true;
    }
    return NextResponse.next();
  }

  // Demo mode lives in localStorage (invisible to middleware); the client
  // layer handles it. Session cookies are the source of truth here.
  const requestHeaders = new Headers(request.headers);
  requestHeaders.delete("x-user-role");
  requestHeaders.delete("x-user-id");
  let response = NextResponse.next({ request: { headers: requestHeaders } });

  const supabase = createServerClient(url, key, {
    cookies: {
      getAll() {
        return request.cookies.getAll();
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value));
        response = NextResponse.next({ request: { headers: requestHeaders } });
        // httpOnly session cookies; refreshed tokens are written here.
        cookiesToSet.forEach(({ name, value, options }) =>
          response.cookies.set(name, value, { ...options, httpOnly: true })
        );
      },
    },
  });

  // getUser() validates the JWT against Supabase (rejects expired/forged)
  // and triggers a refresh via the refresh token when the access token expired.
  let user = null;
  try {
    const { data } = await supabase.auth.getUser();
    user = data.user;
  } catch {
    user = null;
  }

  if (!user) {
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.search = "";
    loginUrl.searchParams.set("next", pathname + request.nextUrl.search);
    return NextResponse.redirect(loginUrl);
  }

  const role = (user.app_metadata?.role as string | undefined) ?? "user";
  requestHeaders.set("x-user-role", role);
  requestHeaders.set("x-user-id", user.id);
  const withRole = NextResponse.next({ request: { headers: requestHeaders } });
  response.cookies.getAll().forEach((c) => withRole.cookies.set(c));
  return withRole;
}

export const config = {
  matcher: ["/dashboard/:path*"],
};
