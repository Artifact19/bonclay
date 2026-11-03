"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase";
import { useAuth } from "@/components/AuthProvider";
import { Button } from "@/components/ui";
import { AuthShell, Divider, Field, FormError, GoogleButton, useNextPath } from "../_components/auth-ui";

function RegisterForm() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const next = useNextPath();
  const supabase = createClient();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (!loading && user) router.replace(next);
  }, [user, loading, router, next]);

  const loginHref = next === "/dashboard" ? "/login" : `/login?next=${encodeURIComponent(next)}`;

  const handleRegister = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: { display_name: displayName },
        emailRedirectTo: `${window.location.origin}${next}`,
      },
    });
    if (error) setError(error.message);
    else setSuccess(true);
    setSubmitting(false);
  };

  const handleGoogleLogin = async () => {
    setError("");
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}${next}` },
    });
    if (error) setError(error.message);
  };

  return (
    <AuthShell subtitle="Create your account">
      {success ? (
        <div role="status" className="space-y-3 py-2 text-center">
          <p className="text-sm font-medium text-accent-success">Check your email for a confirmation link.</p>
          <Link
            href={loginHref}
            className="text-sm font-medium text-accent-secondary underline underline-offset-4 hover:text-text-primary"
          >
            Back to sign in
          </Link>
        </div>
      ) : (
        <>
          <GoogleButton onClick={handleGoogleLogin} />
          <Divider />
          <form onSubmit={handleRegister} className="space-y-4">
            <Field
              id="name"
              label="Display name"
              type="text"
              autoComplete="name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
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
              label="Password (6+ characters)"
              type="password"
              autoComplete="new-password"
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <FormError message={error} />
            <Button type="submit" className="w-full" loading={submitting}>
              Create account
            </Button>
          </form>
          <p className="text-center text-sm text-text-secondary">
            Already registered?{" "}
            <Link href={loginHref} className="font-medium text-accent-secondary underline underline-offset-4 hover:text-text-primary">
              Sign in
            </Link>
          </p>
        </>
      )}
    </AuthShell>
  );
}

export default function RegisterPage() {
  return (
    <Suspense fallback={null}>
      <RegisterForm />
    </Suspense>
  );
}
