/**
 * Route entrance for the app: a pure CSS animation, identical markup on server and client. It sits inside the app
 * shell's LayoutWrapper, so the sidebar (fixed) is never inside a transformed element. The public site has no entrance
 * animation (its pages orchestrate their own motion), so this template is not used there.
 */
export default function Template({ children }: { children: React.ReactNode }) {
  return <div className="page-in">{children}</div>;
}
