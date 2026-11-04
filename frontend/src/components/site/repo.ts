/**
 * The project's GitHub repository is private for now, so the site must not send visitors to it (they would get a 404) or
 * say "the source is on GitHub". One switch controls every such link and sentence: set NEXT_PUBLIC_REPO_PUBLIC=1 (in
 * frontend/.env.local, then rebuild) once the repository is public, and they all come back. Do not turn it on before the
 * git history has been cleaned.
 */
export const REPO_PUBLIC = process.env.NEXT_PUBLIC_REPO_PUBLIC === "1";
export const REPO_URL = "https://github.com/Artifact19/bonclay";
/** A file in the repository, for example repoFile("docs/RESULTS.md"). */
export const repoFile = (path: string) => `${REPO_URL}/blob/main/${path}`;
