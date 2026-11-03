import { Suspense } from "react";
import { VoiceLibrary } from "@/components/voices/VoiceLibrary";

export default function VoicesPage() {
  return (
    <Suspense fallback={null}>
      <VoiceLibrary />
    </Suspense>
  );
}
