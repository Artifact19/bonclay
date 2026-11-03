'use client';

import { Cloud, HardDrive } from 'lucide-react';
import { useSyncExternalStore } from 'react';
import { motion } from 'motion/react';

function subscribeMode(cb: () => void) {
  window.addEventListener('storageModeChanged', cb);
  window.addEventListener('storage', cb);
  return () => {
    window.removeEventListener('storageModeChanged', cb);
    window.removeEventListener('storage', cb);
  };
}
const readMode = () => localStorage.getItem('bonclay_storage_mode') !== 'cloud';

export function StorageToggle() {
  const mounted = useSyncExternalStore(() => () => {}, () => true, () => false);
  const isLocal = useSyncExternalStore(subscribeMode, readMode, () => true);

  const toggleMode = (local: boolean) => {
    localStorage.setItem('bonclay_storage_mode', local ? 'local' : 'cloud');
    // Dispatch a custom event so other components can instantly re-render if needed
    window.dispatchEvent(new Event('storageModeChanged'));
  };

  if (!mounted) return <div className="h-10 w-48 bg-white/5 rounded-full animate-pulse" />;

  return (
    <div className="flex items-center p-1 bg-black/40 border border-white/10 rounded-full w-fit">
      <button
        onClick={() => toggleMode(true)}
        className={`relative flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-full transition-colors z-10 ${
          isLocal ? 'text-black' : 'text-white/60 hover:text-white'
        }`}
      >
        {isLocal && (
          <motion.div
            layoutId="storage-toggle"
            className="absolute inset-0 bg-white rounded-full -z-10"
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          />
        )}
        <HardDrive className="w-4 h-4" />
        Local
      </button>

      <button
        onClick={() => toggleMode(false)}
        className={`relative flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-full transition-colors z-10 ${
          !isLocal ? 'text-black' : 'text-white/60 hover:text-white'
        }`}
      >
        {!isLocal && (
          <motion.div
            layoutId="storage-toggle"
            className="absolute inset-0 bg-[hsl(var(--au-accent))] rounded-full -z-10"
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
          />
        )}
        <Cloud className="w-4 h-4" />
        Cloud
      </button>
    </div>
  );
}
