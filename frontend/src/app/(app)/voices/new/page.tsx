import { redirect } from "next/navigation";

export default function NewVoiceRedirect() {
  redirect("/dashboard/voices?new=1");
}
