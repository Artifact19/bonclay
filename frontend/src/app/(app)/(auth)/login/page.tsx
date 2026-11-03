"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase";
import { useAuth } from "@/components/AuthProvider";
import { Button } from "@/components/ui";
import { AuthShell, Divider, Field, FormError, GoogleButton, useNextPath } from "../_components/auth-ui";

function LoginForm() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const next = useNextPath();
  const supabase = createClient();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [demoing, setDemoing] = useState(false);

  useEffect(() => {
    if (!loading && user) router.replace(next);
  }, [user, loading, router, next]);

  const handleEmailLogin = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      setError(error.message);
      setSubmitting(false);
    }
  };

  const handleGoogleLogin = async () => {
    setError("");
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}${next}` },
    });
    if (error) setError(error.message);
  };

  const handleDemoLogin = () => {
    setError("");
    setDemoing(true);
    // Bypass Supabase entirely for local demo testing
    localStorage.setItem("bonclay_demo_mode", "true");
    window.location.assign(next);
  };

  return (
    <AuthShell subtitle="Sign in to your studio">
      <GoogleButton onClick={handleGoogleLogin} />
      <Button variant="outline" className="w-full" loading={demoing} onClick={handleDemoLogin}>
        Try the demo studio
      </Button>
      <Divider />
      <form onSubmit={handleEmailLogin} className="space-y-4">
        <Field
          id="email"
          label="Email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <Field
          id="password"
          label="Password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <FormError message={error} />
        <Button type="submit" className="w-full" loading={submitting}>
          Sign in
        </Button>
      </form>
      <p className="text-center text-sm text-text-secondary">
        No account?{" "}
        <Link
          href={next === "/dashboard" ? "/register" : `/register?next=${encodeURIComponent(next)}`}
          className="font-medium text-accent-secondary underline underline-offset-4 hover:text-text-primary"
        >
          Create one
        </Link>
      </p>
    </AuthShell>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}
